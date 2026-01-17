import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os

from utils.ReplayBuffer import ReplayBuffer
from agent.model import RSSMEncoder, RewardModel, ContinueModel, RSSMState
from agent.decoder import Decoder
from agent.actor import Actor
from agent.critic import Critic
from utils.video_recorder import VideoRecorder

def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size
    top = (h - new_h)//2
    left = (w - new_w)//2
    image = image[:, top:top + new_h, left:left + new_w]
    return image


def compute_lambda_values(rewards, values, continues, horizon_length, device, lambda_):
    """
    Compute TD(λ) returns for imagined trajectories using tools.py implementation
    """
    rewards = rewards[:-1]
    continues = continues[:-1]
    next_values = values[1:]
    last = next_values[-1]
    
    # One-step TD targets with (1-λ) weighting
    inputs = rewards + continues * next_values * (1 - lambda_)

    outputs = []
    # Iterate backwards through time
    for index in reversed(range(horizon_length - 1)):
        last = inputs[index] + continues[index] * lambda_ * last
        outputs.append(last)
    
    returns = torch.stack(list(reversed(outputs)), dim=0).to(device)
    return returns


def compute_percentile_return_scale(returns, percentile_low=5, percentile_high=95, ema_decay=0.99):
    """
    Compute return normalization scale using percentiles (DreamerV3 technique)
    Returns scale S for normalizing returns
    """
    # Flatten returns for percentile computation
    flat_returns = returns.reshape(-1)
    
    # Compute percentiles
    low = torch.quantile(flat_returns, percentile_low / 100.0)
    high = torch.quantile(flat_returns, percentile_high / 100.0)
    
    scale = high - low
    return scale


class Dreamer:
    def __init__(
        self,
        observation_shape,
        discrete_action_bool,
        action_size,
        writer,
        device,
        config,
        work_dir
    ):
        self.device = device
        self.action_size = action_size
        self.discrete_action_bool = discrete_action_bool
        self.config = config
        self.image_size = observation_shape[-1]
        self.writer = writer
        self.work_dir = work_dir
        
        # Video recording setup
        video_dir = os.path.join(work_dir, 'video')
        os.makedirs(video_dir, exist_ok=True)
        self.video = VideoRecorder(dir_name=video_dir, height=480, width=640)

        # Initialize models with DreamerV3 enhancements
        self.rssm = RSSMEncoder(actions_size=(action_size,),device=self.device).to(self.device)
        self.reward_predictor = RewardModel(config, use_twohot=True).to(self.device)

        self.buffer = ReplayBuffer((3,100,100), 
                                   (action_size,),
                                    device=self.device, 
                                    episode_len=config['environment']['total_frames'],
                                    batch_size=config['parameters']['dreamer']['batch_size'])
        
        self.decoder = Decoder(observation_shape).to(self.device)
        self.actor = Actor(False, action_size).to(self.device)
        self.critic = Critic(use_twohot=True).to(self.device)

        # Optimizer with gradient clipping
        self.model_params = (
            list(self.rssm.parameters())
            + list(self.decoder.parameters())
            + list(self.reward_predictor.parameters())
        )
        self.model_optimizer = torch.optim.Adam(
            self.model_params, lr=self.config['parameters']['dreamer']['model_learning_rate']
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config['parameters']['dreamer']['actor_learning_rate']
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config['parameters']['dreamer']['critic_learning_rate']
        )

        self.logger = writer
        self.num_total_episode = 0
        self.global_step = 0
        
        # Return normalization tracking (EMA)
        self.return_scale = 1.0
        self.ema_decay = 0.99
        
        # Best model tracking
        self.max_mean_eval_reward = -float('inf')

    def train(self, env, eval_env=None):
        """Main training loop with periodic evaluation"""
        if len(self.buffer) < 1:
            print("Collecting initial data...")
            self.environment_interaction(env, 1, train=True)

        eval_freq = self.config['parameters']['dreamer'].get('eval_freq', 5000)
        num_eval_episodes = self.config['parameters']['dreamer'].get('num_eval_episodes', 3)
        save_freq = self.config['parameters']['dreamer'].get('save_freq', 10000)
        
        for iteration in tqdm(range(self.config['parameters']['dreamer']['train_iterations']),
                       desc="Training Iterations", position=0):
            
            # Training step
            for collect_interval in range(1):
                data = self.buffer.sample(
                    self.config['parameters']['dreamer']['batch_size'], 
                    self.config['parameters']['dreamer']['batch_length']
                )
                stochastics, deterministics = self.dynamic_learning(data)
                self.behavior_learning(stochastics, deterministics)
                self.global_step += 1
            
            # Periodic evaluation
            if eval_env is not None and self.global_step % eval_freq == 0 and self.global_step > 0:
                print(f"\nEvaluating at step {self.global_step}...")
                mean_reward = self.evaluate(eval_env, num_eval_episodes)
                
                # Save best model
                if mean_reward > self.max_mean_eval_reward:
                    self.max_mean_eval_reward = mean_reward
                    print(f"New best model! Mean reward: {mean_reward:.2f}")
                    self.save(self.work_dir, self.global_step, episode=self.num_total_episode, is_best=True)
            
            # Periodic saving
            if self.global_step % save_freq == 0 and self.global_step > 0:
                self.save(self.work_dir, self.global_step, episode=self.num_total_episode)
            
            # Collect more environment data periodically
            if (iteration + 1) % 10 == 0:
                self.environment_interaction(env, 1, train=True)

    def evaluate(self, env, num_episodes=5):
        """Evaluate the agent and return mean reward"""
        all_ep_rewards = []
        
        for ep in range(num_episodes):
            prev_state = None
            prev_action = None
            obs_raw, _ = env.reset()
            
            score = 0
            terminated = False
            
            # Enable video recording for first episode
            self.video.init(enabled=(ep == 0))
            
            while not terminated:
                if obs_raw.shape[-1] != self.image_size:
                    obs = center_crop_image(obs_raw, self.image_size)
                else:
                    obs = obs_raw
                
                obs = torch.FloatTensor(obs).to(self.device)
                obs = obs.unsqueeze(0)
                
                # Get action without exploration noise
                with torch.no_grad():
                    state = self.rssm(
                        obs=obs,
                        prev_action=prev_action,
                        prev_state=prev_state
                    )
                    
                    posterior = state.stoch
                    deterministic = state.det
                    
                    action = self.actor(posterior, deterministic, deterministic_action=True)
                    env_action = action.cpu().numpy()[0]

                next_obs, reward, terminated, truncated, info = env.step(env_action)
                
                # Record video for first episode
                if ep == 0:
                    self.video.record(env)
                
                score += reward
                obs_raw = next_obs
                prev_state = state
                prev_action = action

                if terminated or truncated:
                    break
            
            all_ep_rewards.append(score)
            
            # Save video for first episode
            if ep == 0:
                self.video.save(f'eval_step_{self.global_step}.mp4')
        
        # Log evaluation metrics
        mean_reward = np.mean(all_ep_rewards)
        std_reward = np.std(all_ep_rewards)
        min_reward = np.min(all_ep_rewards)
        max_reward = np.max(all_ep_rewards)
        
        self.writer.add_scalar("eval/mean_reward", mean_reward, self.global_step)
        self.writer.add_scalar("eval/std_reward", std_reward, self.global_step)
        self.writer.add_scalar("eval/min_reward", min_reward, self.global_step)
        self.writer.add_scalar("eval/max_reward", max_reward, self.global_step)
        
        print(f"\nEvaluation Results (Step {self.global_step}):")
        print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Min/Max: {min_reward:.2f} / {max_reward:.2f}")
        
        return mean_reward

    def save(self, path, step, episode=0, is_best=False):
        """Save model checkpoint"""
        model_dir = os.path.join(path, 'model')
        os.makedirs(model_dir, exist_ok=True)
        
        checkpoint = {
            'step': step,
            'episode': episode,
            'rssm_state_dict': self.rssm.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'reward_predictor_state_dict': self.reward_predictor.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'model_optimizer_state_dict': self.model_optimizer.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'return_scale': self.return_scale,
            'max_mean_eval_reward': self.max_mean_eval_reward,
            'num_total_episode': self.num_total_episode,
        }
        
        if is_best:
            save_path = os.path.join(model_dir, 'best_model.pt')
            print(f"Saving best model to {save_path}")
        else:
            save_path = os.path.join(model_dir, f'checkpoint_{step}.pt')
            print(f"Saving checkpoint to {save_path}")
        
        torch.save(checkpoint, save_path)
        
        # Also save as latest
        latest_path = os.path.join(model_dir, 'latest_model.pt')
        torch.save(checkpoint, latest_path)
        
        # Save buffer
        buffer_dir = os.path.join(path, 'buffer')
        os.makedirs(buffer_dir, exist_ok=True)
        self.buffer.save(buffer_dir)

    def load(self, path, load_best=False):
        """Load model checkpoint"""
        model_dir = os.path.join(path, 'model')
        
        if load_best:
            checkpoint_path = os.path.join(model_dir, 'best_model.pt')
        else:
            checkpoint_path = os.path.join(model_dir, 'latest_model.pt')
        
        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}")
            return 0
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.rssm.load_state_dict(checkpoint['rssm_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.reward_predictor.load_state_dict(checkpoint['reward_predictor_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        self.model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.return_scale = checkpoint.get('return_scale', 1.0)
        self.max_mean_eval_reward = checkpoint.get('max_mean_eval_reward', -float('inf'))
        self.num_total_episode = checkpoint.get('num_total_episode', 0)
        self.global_step = checkpoint['step']
        
        # Load buffer if exists
        buffer_dir = os.path.join(path, 'buffer')
        if os.path.exists(os.path.join(buffer_dir, 'replay_buffer.npz')):
            self.buffer.load(buffer_dir)
            print("Loaded replay buffer")
        
        print(f"Loaded checkpoint from step {checkpoint['step']}, episode {checkpoint['episode']}")
        return checkpoint['step']

    def dynamic_learning(self, data):
        prior, posterior = self.rssm.encode_sequence(data['observations'], data['actions'])

        self.dynamic_learning_infos = dict({
            'priors_stoch': prior.stoch, 
            'prior_dist_means': prior.mean,
            'prior_dist_stds': prior.std,
            'posteriors_stoch': posterior.stoch,
            'posterior_dist_means': posterior.mean,
            'posterior_dist_stds': posterior.std,
            'posterior_det': posterior.det,
        })
        
        self._model_update(data, self.dynamic_learning_infos)
        return self.dynamic_learning_infos['posteriors_stoch'].detach(), \
               self.dynamic_learning_infos['posterior_det'].detach()

    def _model_update(self, data, posterior_info):
        # 1. Reconstruction loss
        reconstructed_observation_dist = self.decoder(
            posterior_info['posteriors_stoch'][:-1],
            posterior_info['posterior_det'][:-1]
        )
        reconstruction_observation_loss = reconstructed_observation_dist.log_prob(
            data['observations'][1:]
        )
        
        # 2. Reward prediction loss
        reward_dist = self.reward_predictor(
            posterior_info['posteriors_stoch'][:-1],  
            posterior_info['posterior_det'][:-1] 
        )
        reward_loss = reward_dist.log_prob(data['rewards'][1:].unsqueeze(-1))

        # 3. KL divergence loss with free bits
        prior_dist = torch.distributions.Independent(
            torch.distributions.Normal(
                posterior_info['prior_dist_means'],
                posterior_info['prior_dist_stds']
            ),
            1
        )
        
        posterior_dist = torch.distributions.Independent(
            torch.distributions.Normal(
                posterior_info['posterior_dist_means'],
                posterior_info['posterior_dist_stds']
            ),
            1
        )
        
        # KL divergence with free bits
        kl_divergence_loss = torch.distributions.kl.kl_divergence(
            posterior_dist, prior_dist
        )
        
        # Apply free bits
        free_nats = self.config['parameters']['dreamer']['free_nats']
        kl_divergence_loss = torch.max(
            torch.tensor(free_nats).to(self.device), kl_divergence_loss
        ).mean()

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
        
        # Log losses
        if self.global_step % 100 == 0:
            self.writer.add_scalar("train/model_loss", model_loss.item(), self.global_step)
            self.writer.add_scalar("train/kl_loss", kl_divergence_loss.item(), self.global_step)
            self.writer.add_scalar("train/reconstruction_loss", -reconstruction_observation_loss.mean().item(), self.global_step)
            self.writer.add_scalar("train/reward_loss", -reward_loss.mean().item(), self.global_step)

    def behavior_learning(self, stochastics, deterministics):
        stochastic = stochastics.reshape(-1, self.config['parameters']['dreamer']['stochastic_size'])
        deterministic = deterministics.reshape(-1, self.config['parameters']['dreamer']['deterministic_size'])
        
        initial_state = RSSMState(
            mean=torch.zeros_like(stochastic),
            std=torch.ones_like(stochastic),
            stoch=stochastic,
            det=deterministic
        )

        imagined_trajectory = self.rssm.rollout.rollout_imagination(
            self.config['parameters']['dreamer']['horizon_length'],
            self.actor,
            initial_state
        )
        self._agent_update(imagined_trajectory)

    def _agent_update(self, imagined_trajectory):
        """Update actor and critic with return normalization"""
        # Compute predicted rewards - FIXED: added () to call the method
        predicted_rewards = self.reward_predictor(
            imagined_trajectory.stoch, imagined_trajectory.det
        ).mean()
        
        # Compute values - FIXED: added () to call the method
        values = self.critic(
            imagined_trajectory.stoch, imagined_trajectory.det
        ).mean()
        
        continues = self.config['parameters']['dreamer']['discount'] * torch.ones_like(values)

        # Compute lambda returns
        lambda_values = compute_lambda_values(
            predicted_rewards,
            values,
            continues,
            self.config['parameters']['dreamer']['horizon_length'],
            self.device,
            self.config['parameters']['dreamer']['lambda_'],
        )

        # Update return scale with EMA
        current_scale = compute_percentile_return_scale(lambda_values)
        self.return_scale = self.ema_decay * self.return_scale + (1 - self.ema_decay) * current_scale
        
        # Normalize returns
        scale_limit = 1.0
        normalized_scale = max(scale_limit, self.return_scale.item())
        
        # Update actor
        actor_loss = -torch.mean(lambda_values / normalized_scale)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.config['parameters']['dreamer']['clip_grad'],
            norm_type=self.config['parameters']['dreamer']['grad_norm_type'],
        )
        self.actor_optimizer.step()

        # Update critic
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
        
        # Log losses
        if self.global_step % 100 == 0:
            self.writer.add_scalar("train/actor_loss", actor_loss.item(), self.global_step)
            self.writer.add_scalar("train/value_loss", value_loss.item(), self.global_step)
            self.writer.add_scalar("train/return_scale", self.return_scale.item(), self.global_step)

    @torch.no_grad()
    def environment_interaction(self, env, num_interaction_episodes, train=True):
        episode_pbar = tqdm(range(num_interaction_episodes), desc="Environment Interaction", leave=False)

        for ep in episode_pbar:
            prev_state = None
            prev_action = None

            obs_raw, _ = env.reset()
            
            score = 0
            terminated = False

            while not terminated:
                if obs_raw.shape[-1] != self.image_size:
                    obs = center_crop_image(obs_raw, self.image_size)
                else:
                    obs = obs_raw
                
                obs = torch.FloatTensor(obs).to(self.device)
                obs = obs.unsqueeze(0)
                
                state = self.rssm(
                    obs=obs,
                    prev_action=prev_action,
                    prev_state=prev_state
                )
                
                posterior = state.stoch
                deterministic = state.det
                
                action = self.actor(posterior, deterministic)
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

                if terminated or truncated:
                    if train:
                        self.num_total_episode += 1
                        self.writer.add_scalar(
                            "train/episode_reward", score, self.num_total_episode
                        )
                    break