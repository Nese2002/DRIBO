import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import os

from .RSSM import RSSMEncoder
from .SAC import Actor, Critic
from .DRIBO import DRIBO

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )

def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image

class ExponentialScheduler(object):
    def __init__(self, start_value, end_value, n_iterations, start_iteration=0):
        self.log_start = np.log10(start_value)
        self.log_end = np.log10(end_value)
        self.n_iterations = n_iterations
        self.start_iteration = start_iteration
        self.start_value = start_value
        self.end_value = end_value

    def __call__(self, t):
        if t <= self.start_iteration:
            return self.start_value
        elif t > self.start_iteration + self.n_iterations:
            return self.end_value
        else:
            step = t - self.start_iteration
            alpha = step / self.n_iterations
            log_value = self.log_start + alpha * (self.log_end - self.log_start)
            return np.power(10., log_value)
        

class DRIBOSacAgent(object):
    def __init__(
        self,
        #environment
        obses_shape,
        actions_shape,
        #general
        device,
        hidden_dim=512,
        log_interval=100, 
        #critic
        discount=0.99,
        init_temperature=0.1,
        critic_lr=1e-5,
        critic_tau=0.01,
        critic_target_update_freq=2,
        alpha_lr=1e-4,
        alpha_beta=0.5,
        #actor
        actor_lr=1e-5,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        #rssm
        obs_encoder_feature_dim=512,
        stochastic_size=30,
        deterministic_size=200,
        encoder_lr=1e-5,
        encoder_tau=0.05,
        num_layers=4,
        num_filters=16, #16
        #DRIBO
        mib_update_freq=1,
        mib_batch_size=8,
        mib_seq_len=32,
        beta_start_value=1e-4,
        beta_end_value=1e-2,
        grad_clip=500,#
        kl_balancing=True
    ):
        self.device = device
        self.log_interval = log_interval
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.mib_update_freq = mib_update_freq
        self.image_size = obses_shape[-1]
        self.batch_size = mib_batch_size
        self.seq_len = mib_seq_len
        self.grad_clip = grad_clip
        self.kl_balancing = kl_balancing

        #rssm
        self.encoder = RSSMEncoder( obses_shape, actions_shape, obs_encoder_feature_dim,
            stochastic_size, deterministic_size, num_layers,num_filters, hidden_dim, device=device)
        
        self.encoder_target = RSSMEncoder( obses_shape, actions_shape, obs_encoder_feature_dim,
            stochastic_size, deterministic_size, num_layers,num_filters, hidden_dim, device=device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())

        feature_dim = stochastic_size + deterministic_size

        #sac
        self.actor = Actor(actions_shape, hidden_dim, feature_dim,actor_log_std_min, actor_log_std_max).to(device)

        self.critic = Critic(actions_shape, hidden_dim, feature_dim).to(device)

        self.critic_target = Critic(actions_shape, hidden_dim, feature_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(actions_shape)

        # DRIBO
        self.DRIBO = DRIBO(obses_shape, feature_dim,self.encoder, self.encoder_target, self.device).to(device)

        #optimizers
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr, fused=True)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, fused=True)

        self.critic_optimizer = torch.optim.Adam([{'params': self.critic.parameters()},{'params': self.encoder.parameters()}],lr=critic_lr, fused=True)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999), fused=True)

        self.DRIBO_optimizer = torch.optim.Adam(self.DRIBO.parameters(), lr=encoder_lr, fused=True) #self.DRIBO.encoder.parameters()

        self.beta_scheduler = ExponentialScheduler(start_value=beta_start_value, end_value=beta_end_value,n_iterations=5e4, start_iteration=10000)
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        #training
        self.train()
        self.critic_target.train()
        self.encoder_target.train()
        
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.encoder.train(training)
        self.DRIBO.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    #action selection
    def select_action(self, obs, prev_action, prev_state):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            prev_state = self.encoder(obs, prev_action, prev_state)
            latent_state = torch.cat([prev_state.stoch, prev_state.det], dim=-1)

            # latent_state = self.encoder.ln(latent_state)

            mu, _, _, _ = self.actor(latent_state, compute_pi=False, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten(), mu, prev_state
    
    def sample_action(self, obs, prev_action, prev_state):
        if obs.shape[-1] != self.image_size:
            obs = center_crop_image(obs, self.image_size)

        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            current_state = self.encoder(obs, prev_action, prev_state)
            latent_state = torch.cat([current_state.stoch, current_state.det], dim=-1)

            # latent_state = self.encoder.ln(latent_state)

            mu, pi, _, _ = self.actor(latent_state, compute_log_pi=False)
            action = pi.squeeze(0).cpu().numpy()
            
            return action, pi, current_state


    #update
    def update(self, replay_buffer, logger, t):
        obses, positives, actions, rewards, not_done = replay_buffer.sample_multi_view(self.batch_size, self.seq_len)
        
        if t % self.log_interval == 0:
            logger.log('train/batch_reward', rewards.mean(), t)        

        flat_actions = actions[:-1].reshape((self.seq_len-1) * self.batch_size,-1) #2(48,6)
        rewards = rewards[:-1].reshape((self.seq_len-1) * self.batch_size,-1)
        not_done = not_done[:-1].reshape((self.seq_len-1) * self.batch_size,-1)

        #latent states
        _, post = self.DRIBO.encode(obses, actions)
        feature = torch.cat([post.stoch, post.det], dim=-1) #[32,8,230]

        latent_states = feature[:-1].reshape((self.seq_len-1) * self.batch_size,-1).detach()
        q_latent_states = feature[:-1].reshape((self.seq_len-1) * self.batch_size,-1)
        next_latent_states = feature[1:].reshape((self.seq_len-1) * self.batch_size,-1).detach()
        #target latent states
        _, target_post = self.DRIBO.encode(obses, actions, ema=True)
        target_feature = torch.cat([target_post.stoch, target_post.det], dim=-1)
        target_next_latent_states = target_feature[1:].reshape((self.seq_len-1) * self.batch_size,-1).detach()

        self.update_critic(q_latent_states, flat_actions, rewards, next_latent_states, target_next_latent_states, not_done, logger, t)

        if t % self.actor_update_freq == 0:
            self.update_actor_and_alpha(latent_states, logger, t)

        if t % self.critic_target_update_freq == 0:
            soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
            soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
            soft_update_params(self.encoder, self.encoder_target, self.encoder_tau)

        if t % self.mib_update_freq == 0:
            self.update_mib(obses, positives, actions, logger, t)


    def update_critic(self, q_latent_states, flat_actions, rewards, next_latent_states, target_next_latent_states, not_done, logger, t):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_latent_states)
            target_Q1, target_Q2 = self.critic_target(target_next_latent_states, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = rewards + (not_done * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(q_latent_states, flat_actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        if t % self.log_interval == 0:
            logger.log('train_critic/loss', critic_loss, t)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip)

        self.critic_optimizer.step()

        self.critic.log(logger, t, log_interval=self.log_interval)

    def update_actor_and_alpha(self, latent_states, logger, t):
        _, pi, log_pi, log_std = self.actor(latent_states)
        actor_Q1, actor_Q2 = self.critic(latent_states, pi)

        # detach encoder, so we don't update it with the actor loss
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if t % self.log_interval == 0:
            logger.log('train_actor/loss', actor_loss, t)
            logger.log('train_actor/target_entropy', self.target_entropy, t)

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if t % self.log_interval == 0:
            logger.log('train_actor/entropy', entropy.mean(), t)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, t, log_interval=self.log_interval)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

        if t % self.log_interval == 0:
            logger.log('train_alpha/loss', alpha_loss, t)
            logger.log('train_alpha/value', self.alpha, t)

        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_mib(self, obses1, obses2, actions, logger, t):
        seq_len, batch_size, ch, h, w  = obses1.size()
        s1_prior, s1 = self.DRIBO.encode(obses1, actions)
        s2_prior, s2 = self.DRIBO.encode(obses2, actions, ema=True)

        # Maximize mutual information of task-relevant features
        latent_states1 =torch.cat([s1.stoch, s1.det], dim=-1).reshape(seq_len * batch_size, -1)
        latent_states2 = torch.cat([s2.stoch, s2.det], dim=-1).reshape(seq_len * batch_size, -1)

        # latent_states1 = self.encoder.ln(latent_states1)
        # latent_states2 = self.encoder.ln(latent_states2)

        logits = self.DRIBO.compute_logits(latent_states1, latent_states2)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)

        # Minimize the task-irrelevant information
        s1_dist = td.independent.Independent(td.Normal(s1.mean, s1.std), 1)
        s2_dist = td.independent.Independent(td.Normal(s2.mean, s2.std), 1)
        skl = self.DRIBO.compute_skl(s1_dist, s2_dist)
        kl_balancing = self.DRIBO.compute_kl_balancing(s1_prior, s1)
        beta = self.beta_scheduler(t)
        loss += beta * kl_balancing
        loss += beta * skl

        self.encoder_optimizer.zero_grad()
        self.DRIBO_optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip)
        
        self.encoder_optimizer.step()
        self.DRIBO_optimizer.step()

        if t % self.log_interval == 0:
            logger.log('train/DRIBO_loss', loss, t)
            logger.log('train/beta', beta, t)
            logger.log('train/skl', skl, t)

    
    def load(self, model_dir, ):
        """Load all network parameters"""
        checkpoints = [f for f in os.listdir(model_dir) if f.startswith('checkpoint_') and f.endswith('.pt')]
        if not checkpoints:
            print("No checkpoint found!")
            return 0
        steps = [int(f.split('_')[1].split('.')[0]) for f in checkpoints]
        step = max(steps)
        
        checkpoint_path = os.path.join(model_dir, f'checkpoint_{step}.pt')
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load networks
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.encoder_target.load_state_dict(checkpoint['encoder_target'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.DRIBO.load_state_dict(checkpoint['DRIBO'])
        
        # Load optimizers
        self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])
        self.DRIBO_optimizer.load_state_dict(checkpoint['DRIBO_optimizer'])
        
        # Load other parameters
        self.log_alpha = checkpoint['log_alpha'].to(self.device)
        self.log_alpha.requires_grad = True
        
        print(f"Loaded checkpoint from step {step}")
        return checkpoint['step']

    def save(self, model_dir, t, episode):
        """Save all network parameters"""
        torch.save({
            # Networks
            'encoder': self.encoder.state_dict(),
            'encoder_target': self.encoder_target.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'DRIBO': self.DRIBO.state_dict(),
            
            # Optimizers
            'encoder_optimizer': self.encoder_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha_optimizer': self.log_alpha_optimizer.state_dict(),
            'DRIBO_optimizer': self.DRIBO_optimizer.state_dict(),
            
            # Other parameters
            'log_alpha': self.log_alpha,
            'step': t,
            'episode': episode
        }, os.path.join(model_dir, f'checkpoint_{t}.pt'))