import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np

def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def create_normal_dist(
    x,
    std=None,
    mean_scale=1,
    init_std=0,
    min_std=0.1,
    activation=None,
    event_shape=None,
):
    if std == None:
        mean, std = torch.chunk(x, 2, -1)
        mean = mean / mean_scale
        if activation:
            mean = activation(mean)
        mean = mean_scale * mean
        std = F.softplus(std + init_std) + min_std
    else:
        mean = x
    dist = torch.distributions.Normal(mean, std)
    if event_shape:
        dist = torch.distributions.Independent(dist, event_shape)
    return dist


def stack_states(rssm_states, dim):
    return RSSMState(
        torch.stack([state.mean for state in rssm_states], dim=dim),
        torch.stack([state.std for state in rssm_states], dim=dim),
        torch.stack([state.stoch for state in rssm_states], dim=dim),
        torch.stack([state.det for state in rssm_states], dim=dim),
    )


class RSSMState:
    __slots__ = ("mean", "std", "stoch", "det")

    def __init__(self, mean, std, stoch, det):
        self.mean = mean
        self.std = std
        self.stoch = stoch
        self.det = det

    def to_tuple(self):
        return self.mean, self.std, self.stoch, self.det

# depth : 32
# stride : 2
# kernel_size : 4
# activation : ReLU

class ObservationEncoder(nn.Module):
    def __init__(
        self, depth=32, stride=2, kernel_size=4, shape=(3, 84, 84),
        num_layers=4, obs_encoder_feature_dim=1024, activation=nn.ReLU
    ):
        super().__init__()
        self.shape = shape
        self.stride = stride
        self.depth = depth
        self.num_layers = num_layers
        self.obs_encoder_feature_dim = obs_encoder_feature_dim

        self.convs = nn.Sequential(
            nn.Conv2d(shape[0],depth, kernel_size, stride),
            nn.ReLU(),
            nn.Conv2d(depth,depth*2, kernel_size, stride),
            nn.ReLU(),
            nn.Conv2d(depth*2,depth*4, kernel_size, stride),
            nn.ReLU(),
            nn.Conv2d(depth*4,depth*8, kernel_size, stride),
            nn.ReLU(),
        )
        self.convs.apply(initialize_weights)

        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            dummy_output = self.convs(dummy_input)
            self.embed_size = dummy_output.numel() // dummy_output.shape[0]
        
        self.fc = nn.Linear(self.embed_size, obs_encoder_feature_dim)
        self.ln = nn.LayerNorm(obs_encoder_feature_dim)
    
    def forward(self, obses):
        batch_shape = obses.shape[:-3]
        img_shape = obses.shape[-3:]
        obses = obses.reshape(-1, *img_shape)

        # obses = obses / 255.
        embed = self.convs(obses)
        embed = torch.reshape(embed, (np.prod(batch_shape), -1))
        embed = self.fc(embed)
        embed = self.ln(embed)
        # embed = torch.tanh(embed)
        embed = torch.reshape(embed, (*batch_shape, -1))

        return embed
    
    def spatial_attention(self, obs):
        spatial_softmax = nn.Softmax(1)
        img_shape = obs.shape[-3:]
        gs = [None] * len(self.convs)
        x = obs
        for idx, layer in enumerate(self.convs):
            x = layer(
                x.reshape(-1, *img_shape) / 255.
            ) if idx == 0 else layer(x)
            gs[idx] = x
        gs = [gs_.pow(2).mean(1) for gs_ in gs]
        return [spatial_softmax(
            gs_.view(*gs_.size()[:1], -1)
        ).view_as(gs_) for gs_ in gs]


#recurrent_model:
    # hidden_size : 200
    # activation : ELU

# transition_model : 
    # hidden_size : 200
    # num_layers : 2
    # activation : ELU
    # min_std : 0.1

# representation_model:
    # hidden_size : 200
    # num_layers : 2
    # activation : ELU
    # min_std : 0.1

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
        self, transition_model: RSSMTransition, obs_encoder_feature_dim, action_size,
        stochastic_size=30, deterministic_size=200, hidden_size=200,
        activation=nn.ELU
    ):
        super().__init__()
        self.transition_model = transition_model
        self.obs_encoder_feature_dim = obs_encoder_feature_dim
        self.action_size = action_size
        self.stoch_size = stochastic_size
        self.det_size = deterministic_size
        self.hidden_size = hidden_size
        self.activation = activation

        self.ln_stoch = nn.LayerNorm(stochastic_size)

        self.stoch_fc = nn.Sequential(
            nn.Linear(self.det_size+self.obs_encoder_feature_dim,self.hidden_size),
            self.activation(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size,2*self.stoch_size)
        )

    def initial_state(self, batch_size, device):
        """Initialize the RSSM state with zeros"""
        return RSSMState(
            torch.zeros(batch_size, self.stoch_size, device=device),
            torch.zeros(batch_size, self.stoch_size, device=device),
            torch.zeros(batch_size, self.stoch_size, device=device),
            torch.zeros(batch_size, self.det_size, device=device),
        )

    def forward(self,embed_obs,prev_action,prev_state:RSSMState):
        #prior state
        prior_state = self.transition_model(prev_action, prev_state)
        x = torch.cat([prior_state.det, embed_obs], dim=-1)
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

    def forward(self, seq_len, embed_obses,actions, prev_state: RSSMState):
        return self.rollout_representation(
            seq_len, embed_obses, actions, prev_state
        )
    
    def rollout_representation(self, seq_len, embed_obses, actions, prev_state: RSSMState):
        priors = []
        posteriors = []
        for t in range(seq_len):
            prior_state, posterior_state = self.representation_model(
                embed_obses[t], actions[t], prev_state
            )
            priors.append(prior_state)
            posteriors.append(posterior_state)
            prev_state = posterior_state
        prior = stack_states(priors, dim=0)
        post = stack_states(posteriors, dim=0)
        return prior, post
    
    def rollout_imagination(self, horizon_length, actor, initial_state: RSSMState):
        """
        Imagine future trajectories using transition model only (no observations)
        
        Args:
            horizon_length: Number of steps to imagine
            actor: Policy network to generate actions
            initial_state: Starting RSSMState
        Returns:
            imagined_states: List of imagined RSSMStates
        """

        imagined_priors = []
        prev_state = initial_state
        
        for t in range(horizon_length):
            # Get action from actor policy
            action = actor(prev_state.stoch, prev_state.det)  
            
            # Predict next state using transition model (no observation)
            next_state = self.transition_model(action, prev_state)
            
            imagined_priors.append(next_state)
            prev_state = next_state
        
        # Stack imagined states
        imagined_trajectory = stack_states(imagined_priors, dim=0)
        return imagined_trajectory
    
class RSSMEncoder(nn.Module):
    def __init__(
    self, obs_shape=(3, 84, 84), actions_size=6, obs_encoder_feature_dim=1024,
    stochastic_size=30, deterministic_size=200, num_layers=4, num_filters=32,
    hidden_size=1024, dtype=torch.float, device=None
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.actions_size = self.actions_size = np.prod(actions_size) if isinstance(actions_size, (list, tuple)) else actions_size
        actions_size = self.actions_size = np.prod(actions_size) if isinstance(actions_size, (list, tuple)) else actions_size
        self.stoch_size = stochastic_size
        self.det_size = deterministic_size
        self.latent_size = stochastic_size + deterministic_size
        self.num_layers = num_layers
        self.dtype = dtype
        self.device = device

        # observation encoder
        self.observation_encoder = ObservationEncoder(
            shape=obs_shape, num_layers=num_layers, obs_encoder_feature_dim=obs_encoder_feature_dim,
            depth=num_filters
        )
        # pixel_embed_size = self.observation_encoder.obs_encoder_feature_dim

        # RSSM model
        self.transition = RSSMTransition(
            actions_size, stochastic_size, deterministic_size, hidden_size
        )
        self.representation = RSSMRepresentation(
            self.transition, obs_encoder_feature_dim, actions_size,
            stochastic_size, deterministic_size, hidden_size
        )
        self.rollout = RSSMRollout(self.representation, self.transition)

        # layer normalization
        self.ln = nn.LayerNorm(self.latent_size)


    def forward(self, obs, prev_action, prev_state):
        """
        Single-step inference
        Use for action selection
        """
        state = self.get_state_representation(obs, prev_action, prev_state)
        return state
    
    def get_state_representation(self, obs, prev_action, prev_state):
        # 1. Encode pixels to features
        embed_obses = self.observation_encoder(obs)

        # 2. Initialize if first step
        if prev_action is None:
            prev_action = torch.zeros(obs.size(0), self.actions_size, device=self.device)
        if prev_state is None:
            prev_state = self.representation.initial_state(prev_action.size(0), device=self.device)

        # 3. Infer current state using representation network  
        _, state = self.representation(embed_obses, prev_action, prev_state)
        return state
    
    def encode_sequence(self, obes, actions):
        """
        Sequential encoding for trajectory data 
        Use for representation learning
        """
        seq_len, batch_size, ch, h, w = obes.size()
       
        # Prepare actions with zero padding at start
        prev_actions = actions[:-1]
        prev_action = torch.zeros(batch_size,self.actions_size, device=self.device, dtype=actions.dtype).unsqueeze(0)
        prev_actions = torch.cat([prev_action, prev_actions], dim=0)
        
        # Initialize state
        prev_state = self.representation.initial_state(batch_size, device=self.device)
        
        # Encode all observations at once
        embed_obses = self.observation_encoder(obes)

        # Run rollout through sequence
        prior, posterior = self.rollout.rollout_representation(
            seq_len, embed_obses, prev_actions, prev_state
        )

        return prior, posterior

# hidden_size : 400
# num_layers : 2
# activation : ELU


class RewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config['parameters']['dreamer']
        self.stochastic_size = self.config['stochastic_size']
        self.deterministic_size = self.config['deterministic_size']
 
        self.network = nn.Sequential(
            nn.Linear(self.stochastic_size + self.deterministic_size, 400),
            nn.ELU(),
            nn.Linear(400, 1)
        )
        self.network.apply(initialize_weights)

    def forward(self, posterior, deterministic):
        """
        Args:
            posterior: (seq_len, batch_size, stoch_size)
            deterministic: (seq_len, batch_size, det_size)
        Returns:
            dist: Independent Normal distribution over predicted rewards
        """
        # Get seq_len and batch_size
        seq_len, batch_size = posterior.shape[:2]
        
        # Concatenate posterior and deterministic
        x = torch.cat((posterior, deterministic), dim=-1)
        
        # Flatten seq_len and batch dimensions
        x = x.reshape(seq_len * batch_size, -1)
        
        # Pass through network
        x = self.network(x)
        
        # Reshape back to (seq_len, batch_size, 1)
        x = x.reshape(seq_len, batch_size, 1)
        
        # Create Normal distribution with mean=x, std=1
        # Independent over the scalar reward dimension
        dist = td.Independent(
            td.Normal(x, torch.ones_like(x)), 
            1  # event_shape=1 for scalar reward
        )
        return dist


class ContinueModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config['parameters']['dreamer']
        self.stochastic_size = self.config['stochastic_size']
        self.deterministic_size = self.config['deterministic_size']
 
        self.network = nn.Sequential(
            nn.Linear(self.stochastic_size + self.deterministic_size, 400),
            nn.ELU(),
            nn.Linear(400, 400),
            nn.ELU(),
            nn.Linear(400, 1)
        )
        self.network.apply(initialize_weights)

    def forward(self, posterior, deterministic):
        """
        Args:
            posterior: (seq_len, batch_size, stoch_size)
            deterministic: (seq_len, batch_size, det_size)
        Returns:
            dist: Bernoulli distribution over continue flag
        """
        # Get seq_len and batch_size
        seq_len, batch_size = posterior.shape[:2]
        
        # Concatenate posterior and deterministic
        x = torch.cat((posterior, deterministic), dim=-1)
        
        # Flatten seq_len and batch dimensions
        x = x.reshape(seq_len * batch_size, -1)
        
        # Pass through network
        x = self.network(x)
        
        # Reshape back to (seq_len, batch_size, 1)
        x = x.reshape(seq_len, batch_size, 1)
        
        # Create Bernoulli distribution from logits
        # Squeeze the last dimension for proper Bernoulli shape
        dist = td.Bernoulli(logits=x.squeeze(-1))  # (seq_len, batch_size)
        return dist