import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from RSSM import RSSMEncoder
from SAC import Actor, Critic
import DRIBO

class ExponentialScheduler(object):
    def __init__(self, start_value, end_value, n_iterations, start_iteration=0):
        self.log_start = np.log10(start_value)
        self.log_end = np.log10(end_value)
        self.n_iterations = n_iterations
        self.start_iteration = start_iteration
        self.start_value = start_value
        self.end_value = end_value

    def __call__(self, step):
        if step <= self.start_iteration:
            return self.start_value
        elif step > self.start_iteration + self.n_iterations:
            return self.end_value
        else:
            t = step - self.start_iteration
            alpha = t / self.n_iterations
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
        hidden_dim=256,
        #critic
        discount=0.99,
        init_temperature=0.01,
        critic_lr=1e-3,
        critic_tau=0.005,
        critic_target_update_freq=2,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        #actor
        actor_lr=1e-3,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        #rssm
        obs_encoder_feature_dim=50,
        stochastic_size=30,
        deterministic_size=200,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        num_layers=4,
        num_filters=32,
        #DRIBO
        mib_update_freq=1,
        mib_batch_size=10,
        mib_seq_len=50,
        beta_start_value=1e-3,
        beta_end_value=1,
        grad_clip=500,#
        kl_balancing=True
    ):
        self.device = device
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
        # self.grad_clip = grad_clip
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
        self.DRIBO = DRIBO(obses_shape, feature_dim,self.encoder, self.encoder_target).to(device)

        #optimizers
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_optimizer = torch.optim.Adam([{'params': self.critic.parameters()},{'params': self.encoder.parameters()}],lr=critic_lr)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.DRIBO_optimizer = torch.optim.Adam(self.DRIBO.encoder.parameters(), lr=encoder_lr)

        self.beta_scheduler = ExponentialScheduler(start_value=beta_start_value, end_value=beta_end_value,n_iterations=5e4, start_iteration=10000)
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        #training
        self.critic_target.train()
        self.encoder_target.train()
        self.actor.train()
        self.critic.train()
        self.encoder.train()
        self.DRIBO.train()

    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def update(self, replay_buffer, step):
        obses, actions, reward, not_done = replay_buffer.sample_multi_view(self.batch_size, self.seq_len)

        flat_actions = actions[:-1].reshape((self.seq_len-1) * self.batch_size,-1)
        rewards = reward[:-1].reshape((self.seq_len-1) * self.batch_size,-1)
        not_done = not_done[:-1].reshape((self.seq_len-1) * self.batch_size,-1)

        #latent states
        _, post = self.DRIBO.encode(obses, actions)
        feature = torch.cat([post.stoch, post.det], dim=-1)

        latent_states = feature[:-1].reshape((self.seq_len-1) * self.batch_size,-1).detach()
        q_latent_states = feature[:-1].reshape((self.seq_len-1) * self.batch_size,-1)
        next_latent_states = feature[:-1].reshape((self.seq_len-1) * self.batch_size,-1).detach()

        #target latent states
        _, target_post = self.DRIBO.encode(obses, actions, ema=True)
        target_feature = torch.cat([target_post.stoch, target_post.det], dim=-1)
        target_next_latent_states = target_feature[1:].reshape((self.seq_len-1) * self.batch_size,-1).detach()