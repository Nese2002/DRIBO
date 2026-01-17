import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from utils.ReplayBuffer import ReplayBuffer
from agent.model import RSSMEncoder, RewardModel, ContinueModel, RSSMState
from agent.decoder import Decoder
from agent.actor import Actor
from agent.critic import Critic



def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image

def compute_lambda_values(rewards, values, continues, horizon_length, device, lambda_):
    """
    Compute TD(λ) returns for imagined trajectories
    
    Args:
        rewards: (seq_len, batch_size, 1) - predicted rewards
        values: (seq_len, batch_size, 1) - predicted values
        continues: (seq_len, batch_size, 1) - continuation flags
        horizon_length: int - length of imagination horizon
        device: torch device
        lambda_: float - λ parameter for TD(λ) mixing
    
    Returns:
        returns: (seq_len-1, batch_size, 1) - TD(λ) target values
    """
    # Remove last timestep from rewards and continues (no next value to bootstrap)
    rewards = rewards[:-1]
    continues = continues[:-1]
    
    # Get next step values (offset by 1 timestep)
    next_values = values[1:]
    
    # Start bootstrapping from the last timestep's next value
    last = next_values[-1]
    
    # Compute one-step TD targets with (1-λ) weighting
    inputs = rewards + continues * next_values * (1 - lambda_)

    outputs = []
    # Iterate backwards through time to compute λ-returns
    for index in reversed(range(horizon_length - 1)):
        last = inputs[index] + continues[index] * lambda_ * last
        outputs.append(last)
    
    # Stack outputs in forward time order
    returns = torch.stack(list(reversed(outputs)), dim=0).to(device)
    return returns



class Dreamer:
    def __init__(
        self,
        observation_shape, #3,84,84
        discrete_action_bool,
        action_size,
        writer,
        device,
        config,
    ):
        self.device = device
        self.action_size = action_size
        self.discrete_action_bool = discrete_action_bool
        self.config = config
        self.image_size = observation_shape[-1]
        self.writer = writer

        self.rssm = RSSMEncoder(actions_size=(action_size,)).to(self.device)
        self.reward_predictor = RewardModel(config).to(self.device)

        self.buffer = ReplayBuffer((3,100,100), 
                                   (action_size,),
                                    device=self.device, 
                                    episode_len=config['environment']['total_frames'],
                                    batch_size=config['parameters']['dreamer']['batch_size'])
        self.decoder = Decoder(observation_shape).to(self.device)
        self.actor = Actor(False, action_size).to(self.device)
        self.critic = Critic().to(self.device)

        # optimizer
        self.model_params = (
            list(self.rssm.parameters())
            + list(self.decoder.parameters())
            + list(self.reward_predictor.parameters())
        )
        self.model_optimizer = torch.optim.Adam(
            self.model_params, lr=self.config['parameters']['dreamer']['model_learning_rate'] #0.0006
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config['parameters']['dreamer']['actor_learning_rate']
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config['parameters']['dreamer']['critic_learning_rate']
        )

        self.logger = writer
        self.num_total_episode = 0

    def train(self,env):
        if len(self.buffer) < 1:
            print("collecting data")
            self.environment_interaction(env, 1) #5

        for iteration in tqdm(range(self.config['parameters']['dreamer']['train_iterations']), #1000
                       desc="Iterations", position=0):
            for collect_interval in tqdm(range(self.config['parameters']['dreamer']['collect_interval']), #100
                                  desc=f"Iter {iteration+1}", position=1, leave=False):
                data = self.buffer.sample(
                    self.config['parameters']['dreamer']['batch_size'], self.config['parameters']['dreamer']['batch_length']
                )
                stochastics, deterministics = self.dynamic_learning(data)
                self.behavior_learning(stochastics, deterministics)
    
    
    def evaluate(self, env):
        self.environment_interaction(env, self.config.num_evaluate, train=False)

    def dynamic_learning(self, data):
        prior, posterior = self.rssm.encode_sequence(data['observations'], data['actions'])

        self.dynamic_learning_infos = dict({
            'priors_stoch': prior.stoch, 
            'prior_dist_means': prior.mean, #(seq_len, batch_size)
            'prior_dist_stds': prior.std,
            'posteriors_stoch': posterior.stoch,
            'posterior_dist_means': posterior.mean,
            'posterior_dist_stds': posterior.std,
            'posterior_det': posterior.det,
        })
        
        self._model_update(data, self.dynamic_learning_infos)
        return self.dynamic_learning_infos['posteriors_stoch'].detach(), self.dynamic_learning_infos['posterior_det'].detach()

    def _model_update(self, data, posterior_info): #data=posterior_info=(seq_len,batch_size,...)
        
        # 1. Reconstruction loss
        reconstructed_observation_dist = self.decoder(
            posterior_info['posteriors_stoch'][:-1],  # (seq_len-1, batch_size, stoch_size)
            posterior_info['posterior_det'] [:-1]      # (seq_len-1, batch_size, det_size)
        )
        reconstruction_observation_loss = reconstructed_observation_dist.log_prob(
            data['observations'][1:]  # (seq_len-1, batch_size, C, H, W)
        )
        
        # 2. Reward prediction loss
        reward_dist = self.reward_predictor(
            posterior_info['posteriors_stoch'][:-1],  
            posterior_info['posterior_det'][:-1] 
        )
        
        reward_loss = reward_dist.log_prob(data['rewards'][1:].unsqueeze(-1))

        # 3. KL divergence loss
        # Create prior distribution
        prior_dist = torch.distributions.Independent(
            torch.distributions.Normal(
                posterior_info['prior_dist_means'],  # (seq_len, batch_size, stoch_size)
                posterior_info['prior_dist_stds']
            ),
            1  # event_shape=1 for treating stoch_size as event dimension
        )
        
        # Create posterior distribution  
        posterior_dist = torch.distributions.Independent(
            torch.distributions.Normal(
                posterior_info['posterior_dist_means'],  # (seq_len, batch_size, stoch_size)
                posterior_info['posterior_dist_stds']
            ),
            1  # event_shape=1
        )
        
        # Compute KL divergence
        kl_divergence_loss = torch.distributions.kl.kl_divergence(
            posterior_dist, prior_dist
        ).mean()

        kl_divergence_loss = torch.max(
            torch.tensor(self.config['parameters']['dreamer']['free_nats']).to(self.device), kl_divergence_loss
        )    

        # Total model loss
        model_loss = (
            self.config['parameters']['dreamer']['kl_divergence_scale'] * kl_divergence_loss
            - reconstruction_observation_loss.mean()
            - reward_loss.mean()
        )    

        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            self.model_params,
            self.config['parameters']['dreamer']['clip_grad'],
            norm_type=self.config['parameters']['dreamer']['grad_norm_type'],
        )
        self.model_optimizer.step()


    def behavior_learning(self, stochastics, deterministics):
        """
        #TODO : last posterior truncation(last can be last step)
        posterior shape : (batch, timestep, stochastic)
        """
        stochastic = stochastics.reshape(-1, self.config['parameters']['dreamer']['stochastic_size']) #(seq_len*batch_size, 30)
        deterministic = deterministics.reshape(-1, self.config['parameters']['dreamer']['deterministic_size'])
        
        initial_state = RSSMState(
            mean=torch.zeros_like(stochastic),  # Mean/std not used in imagination
            std=torch.ones_like(stochastic),
            stoch=stochastic,                    # Use ACTUAL stochastic state
            det=deterministic               # Use ACTUAL deterministic state
        )

        imagined_trajectory = self.rssm.rollout.rollout_imagination(
            self.config['parameters']['dreamer']['horizon_length'],
            self.actor,
            initial_state
        )
        self._agent_update(imagined_trajectory)


    def _agent_update(self, imagined_trajectory):
        """
        Update actor and critic using imagined trajectories
        
        Args:
            imagined_trajectory: RSSMState with stacked states
                - stoch: (horizon_length, batch_size, stochastic_size)
                - det: (horizon_length, batch_size, deterministic_size)
        """
        # Compute predicted rewards for all imagined steps
        predicted_rewards = self.reward_predictor(
            imagined_trajectory.stoch, imagined_trajectory.det
        ).mean
        
        # Compute values for all imagined steps
        values = self.critic(
            imagined_trajectory.stoch, imagined_trajectory.det
        ).mean

        
        continues = self.config['parameters']['dreamer']['discount'] * torch.ones_like(values)

        # Compute lambda returns (TD(λ) targets)
        lambda_values = compute_lambda_values(
            predicted_rewards,
            values,
            continues,
            self.config['parameters']['dreamer']['horizon_length'],
            self.device,
            self.config['parameters']['dreamer']['lambda_'],
        )

        # Update actor to maximize lambda returns
        actor_loss = -torch.mean(lambda_values)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.config['parameters']['dreamer']['clip_grad'],
            norm_type=self.config['parameters']['dreamer']['grad_norm_type'],
        )
        self.actor_optimizer.step()

        # Update critic to predict lambda returns
        # Exclude last timestep since it has no bootstrap value
        value_dist = self.critic(
            imagined_trajectory.stoch.detach()[:-1],
            imagined_trajectory.det.detach()[:-1],
        )
        value_loss = -torch.mean(value_dist.log_prob(lambda_values.detach()))

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.config['parameters']['dreamer']['clip_grad'],
            norm_type=self.config['parameters']['dreamer']['grad_norm_type'],
        )
        self.critic_optimizer.step()
      


    @torch.no_grad()
    def environment_interaction(self, env, num_interaction_episodes, train=True):
        episode_pbar = tqdm(range(num_interaction_episodes), desc="Random Sampling")

        for ep in episode_pbar:
            prev_state = None  # Will auto-initialize on first call
            prev_action = None  # Will auto-initialize on first call

            obs_raw,_ = env.reset()
            
            score = 0
            score_lst = np.array([])
            terminated = False

            while not terminated:
                if obs_raw.shape[-1] != self.image_size:
                    obs = center_crop_image(obs_raw, self.image_size)
                
                obs = torch.FloatTensor(obs).to(self.device)
                obs = obs.unsqueeze(0)
                # Call get_state_representation
                # Note: obs should NOT be embedded - the method does that internally
                state = self.rssm(
                    obs=obs,           # Raw observation (not embedded)
                    prev_action=prev_action,   # Previous action
                    prev_state=prev_state      # RSSMState object
                )
                
                # Extract stochastic and deterministic from returned state
                posterior = state.stoch  # or state[2] if it's a tuple
                deterministic = state.det # or state[3] if it's a tuple
                
                # Select action
                action = self.actor(posterior, deterministic).detach()

                env_action = action.cpu().numpy()[0]

                next_obs, reward, terminated, truncated, info = env.step(env_action)
                if train:
                    self.buffer.add(
                        obs_raw, env_action, reward, next_obs, terminated
                    )

                score += reward
                obs_raw = next_obs
                prev_state = state
                prev_action = action

                if terminated:
                    if train:
                        self.num_total_episode += 1
                        self.writer.add_scalar(
                            "training score", score, self.num_total_episode
                        )
                    else:
                        score_lst = np.append(score_lst, score)
                    break
        if not train:
            evaluate_score = score_lst.mean()
            print("evaluate score : ", evaluate_score)
            self.writer.add_scalar("test score", evaluate_score, self.num_total_episode)