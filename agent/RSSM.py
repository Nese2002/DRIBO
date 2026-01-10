import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np

def stack_states(rssm_states: list, dim):
    return RSSMState(
        torch.stack([state.mean for state in rssm_states], dim=dim),
        torch.stack([state.std for state in rssm_states], dim=dim),
        torch.stack([state.stoch for state in rssm_states], dim=dim),
        torch.stack([state.det for state in rssm_states], dim=dim),
    )

def flatten_states(rssm_states, batch_shape):
    return RSSMState(
        torch.reshape(rssm_states.mean, (batch_shape, -1)),
        torch.reshape(rssm_states.std, (batch_shape, -1)),
        torch.reshape(rssm_states.stoch, (batch_shape, -1)),
        torch.reshape(rssm_states.deter, (batch_shape, -1)),
    )


class ObservationEncoder(nn.Module):
    def __init__(
        self, depth=32, stride=1, shape=(3, 84, 84), output_logits=False,
        num_layers=2, obs_encoder_obs_encoder_feature_dim=50, activation=nn.ReLU
    ):
        super().__init__()
        self.shape = shape
        self.stride = stride
        self.depth = depth
        self.num_layers = num_layers
        self.obs_encoder_obs_encoder_feature_dim = obs_encoder_obs_encoder_feature_dim
        self.output_logits = output_logits

        self.convs = nn.Sequential(
            nn.Conv2d(shape[0],depth,3,stride=2),
            nn.ReLU(),
            nn.Conv2d(depth,depth,3,stride=stride),
            nn.ReLU(),
        )

        out_dim = 39
        self.embed_size = depth * out_dim * out_dim
        
        self.fc = nn.Linear(self.embed_size, obs_encoder_obs_encoder_feature_dim)
        self.ln = nn.LayerNorm(obs_encoder_obs_encoder_feature_dim)
    
    def forward(self, obs):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        obs = obs.reshape(-1, *img_shape)

        obs = obs / 255.
        embed = self.convs(obs)
        embed = torch.reshape(embed, (np.prod(batch_shape), -1))
        embed = self.fc(embed)
        embed = self.ln(embed)
        if not self.output_logits:
            embed = torch.tanh(embed)
        embed = torch.reshape(embed, (*batch_shape, -1))
        return embed


class RSSMState:
    __slots__ = ("mean", "std", "stoch", "det")

    def __init__(self, mean, std, stoch, det):
        self.mean = mean
        self.std = std
        self.s = stoch
        self.h = det

    def __getitem__(self, idx):
        return RSSMState(
            self.mean[idx],
            self.std[idx],
            self.stoch[idx],
            self.det[idx],
        )

    def to_tuple(self):
        return self.mean, self.std, self.stoch, self.det

class RSSMTransition(nn.Module):
    def __init__(self, action_size, stochastic_size=30, deterministic_size=200,
        hidden_size=200, activation=nn.ELU
    ):
        super().__init__()
        self.action_size = action_size
        self.stoch_size = stochastic_size
        self.det_size = deterministic_size
        self.hidden_size = hidden_size
        self.activation = activation
        
        self.gru = nn.GRUCell(hidden_size, deterministic_size)
        self.ln_stoch = nn.LayerNorm(stochastic_size)
        self.ln_det = nn.LayerNorm(deterministic_size)

        self.rnn_input_fc = nn.Sequential(
            nn.Linear(self.action_size+self.stoch_size, self.hidden_size),
            self.activation()
        )

        self.stoch_fc = nn.Sequential(
            nn.Linear(self.det_size, self.hidden_size),
            self.activation(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 2 * self.stoch_size),
        )

    def forward(self,prev_action,prev_state:RSSMState):
        x = torch.cat([prev_action, prev_state.stoch], dim=-1)
        rnn_input = self.rnn_input_fc(x)

        #GRU update
        det = self.gru(rnn_input,prev_state.det)
        det = self.ln_det(det)

        #stochastic prior 
        mean,std = torch.chunk(self.stoch_fc(det),2,dim=-1)
        std = F.softplus(std) + 0.1

        #sample stochastic state
        distribution = td.Normal(mean,std)
        stoch = distribution.rsample()
        stoch = self.ln_stoch(stoch)
        return RSSMState(mean,std,stoch,det)

class RSSMRepresentation(nn.Module):
    def __init__(
        self, transition_model: RSSMTransition, obs_encoder_obs_encoder_feature_dim, action_size,
        stochastic_size=30, deterministic_size=200, hidden_size=200,
        activation=nn.ELU
    ):
        super().__init__()
        self.transition_model = transition_model
        self.obs_encoder_obs_encoder_feature_dim = obs_encoder_obs_encoder_feature_dim
        self.action_size = action_size
        self.stoch_size = stochastic_size
        self.det_size = deterministic_size
        self.hidden_size = hidden_size
        self.activation = activation

        self.ln_stoch = nn.LayerNorm(stochastic_size)

        self.stoch_fc = nn.Sequential(
            nn.Linear(self.det_size+self.obs_encoder_obs_encoder_feature_dim,self.hidden_size),
            self.activation(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size,2*self.stoch_size)
        )

    def forward(self,obs_embed,prev_action,prev_state:RSSMState):
        #prior state
        prior_state = self.transition_model(prev_action, prev_state)
        x = torch.cat([prior_state.det, obs_embed], dim=-1)

        #stochastic prior 
        mean,std = torch.chunk(self.stoch_fc(x),2,dim=-1)
        std = F.softplus(std) + 0.1

        #sample stochastic state
        distribution = td.Normal(mean,std)
        stoch = distribution.rsample()
        stoch = self.ln_stoch(stoch)

        #posterior state
        posterior_state = RSSMState(mean,std,stoch,prior_state.det)

        return prior_state, posterior_state
    
class RSSMRollout(nn.Module):
    def __init__(
        self, representation_model: RSSMRepresentation,
        transition_model: RSSMTransition
    ):
        super().__init__()
        self.representation_model = representation_model
        self.transition_model = transition_model

    def forward(self, steps, obs_embed,action, prev_state: RSSMState):
        return self.rollout_representation(
            steps, obs_embed, action, prev_state
        )
    
    def rollout_representation(self, steps, obs_embed, action, prev_state: RSSMState):
        priors = []
        posteriors = []
        for t in range(steps):
            prior_state, posterior_state = self.representation_model(
                obs_embed[t], action[t], prev_state
            )
            priors.append(prior_state)
            posteriors.append(posterior_state)
            prev_state = posterior_state
        prior = stack_states(priors, dim=0)
        post = stack_states(posteriors, dim=0)
        return prior, post
    
class RSSMEncoder(nn.Module):
    def __init__(
    self, obs_shape, action_shape, obs_encoder_feature_dim=50,
    stochastic_size=30, deterministic_size=200, num_layers=2, num_filters=32,
    hidden_size=200, dtype=torch.float, output_logits=False, device=None
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = np.prod(action_shape)
        action_size = np.prod(action_shape)
        self.stoch_size = stochastic_size
        self.det_size = deterministic_size
        self.obs_encoder_feature_dim = stochastic_size + deterministic_size # Check
        self.num_layers = num_layers
        self.dtype = dtype
        self.output_logits = output_logits
        self.device = device

        # observation encoder
        self.observation_encoder = ObservationEncoder(
            shape=obs_shape, num_layers=num_layers, obs_encoder_feature_dim=obs_encoder_feature_dim,
            depth=num_filters, output_logits=self.output_logits
        )
        pixel_embed_size = self.observation_encoder.obs_encoder_feature_dim

        # RSSM model
        self.transition = RSSMTransition(
            action_size, stochastic_size, deterministic_size, hidden_size
        )
        self.representation = RSSMRepresentation(
            self.transition, pixel_embed_size, action_size,
            stochastic_size, deterministic_size, hidden_size
        )
        self.rollout = RSSMRollout(self.representation, self.transition)

        # layer normalization
        self.ln = nn.LayerNorm(self.obs_encoder_feature_dim)

    def initial_state(self, B, device):
        return RSSMState(
            torch.zeros(B, self.stoch_size, device),
            torch.zeros(B, self.stoch_size, device),
            torch.zeros(B, self.stoch_size, device),
            torch.zeros(B, self.det_size, device),
        )

    def forward(self, obs, prev_action=None, prev_state: RSSMState = None):
        # 1. Encode pixels to features
        obs_embed = self.observation_encoder(obs)

        # 2. Initialize if first step
        if prev_action is None:
            prev_action = torch.zeros(
                obs.size(0), self.action_shape, device=self.device
            )
        if prev_state is None:
            prev_state = self.representation.initial_state(
                prev_action.size(0), device=self.device
            )

        # 3. Infer current state using representation network  
        _, state = self.representation(obs_embed, prev_action, prev_state)
        return state