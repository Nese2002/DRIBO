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

# Add symlog/symexp transformations
def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

class OneHotDist(td.OneHotCategorical):
    """OneHot distribution with unimix and straight-through gradients"""
    def __init__(self, logits=None, probs=None, unimix_ratio=0.01):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=()):
        sample = super().sample(sample_shape).detach()
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample

class SymlogDist:
    """Distribution for symlog-transformed predictions"""
    def __init__(self, mode, dist='mse', agg='sum', tol=1e-8):
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape
        if self._dist == 'mse':
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == 'abs':
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == 'mean':
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == 'sum':
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss

class TwoHotDist:
    """Two-hot encoding distribution for robust value/reward prediction"""
    def __init__(self, logits, low=-20.0, high=20.0, device='cuda'):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255, device=device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255

    def mean(self):
        _mean = self.probs * self.buckets
        return symexp(torch.sum(_mean, dim=-1, keepdim=True))

    def mode(self):
        return self.mean()

    def log_prob(self, x):
        x = symlog(x)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)
        return (target * log_pred).sum(-1)


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
            nn.Conv2d(shape[0], depth, kernel_size, stride),
            nn.ReLU(),
            nn.Conv2d(depth, depth*2, kernel_size, stride),
            nn.ReLU(),
            nn.Conv2d(depth*2, depth*4, kernel_size, stride),
            nn.ReLU(),
            nn.Conv2d(depth*4, depth*8, kernel_size, stride),
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

        # Apply symlog transformation for robustness
        obses = symlog(obses / 255.0)
        
        embed = self.convs(obses)
        embed = torch.reshape(embed, (np.prod(batch_shape), -1))
        embed = self.fc(embed)
        embed = self.ln(embed)
        embed = torch.reshape(embed, (*batch_shape, -1))

        return embed

class RSSMTransition(nn.Module):
    def __init__(self, action_size, stochastic_size=30, deterministic_size=200,
        hidden_size=200, activation=nn.ELU, unimix_ratio=0.01
    ):
        super().__init__()
        self.action_size = action_size
        self.stoch_size = stochastic_size
        self.det_size = deterministic_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.unimix_ratio = unimix_ratio
        
        self.gru = nn.GRUCell(hidden_size, deterministic_size)
        self.ln_stoch = nn.LayerNorm(stochastic_size)
        self.ln_det = nn.LayerNorm(deterministic_size)

        self.rnn_input_fc = nn.Sequential(
            nn.Linear(self.action_size + self.stoch_size, self.hidden_size),
            self.activation()
        )

        # Output 32 classes per stochastic dimension (like DreamerV3)
        self.num_classes = 32
        self.stoch_fc = nn.Sequential(
            nn.Linear(self.det_size, self.hidden_size),
            self.activation(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.stoch_size * self.num_classes),
        )

    def forward(self, prev_action, prev_state: RSSMState):
        x = torch.cat([prev_action, prev_state.stoch], dim=-1)
        rnn_input = self.rnn_input_fc(x)

        # GRU update
        det = self.gru(rnn_input, prev_state.det)
        det = self.ln_det(det)

        # Stochastic prior using categorical distribution
        logits = self.stoch_fc(det)
        logits = logits.reshape(*logits.shape[:-1], self.stoch_size, self.num_classes)
        
        # Sample from categorical with unimix
        dist = OneHotDist(logits=logits, unimix_ratio=self.unimix_ratio)
        stoch = dist.sample()
        # Fix: flatten only the one-hot encoding dimension, keeping stoch_size separate
        stoch = stoch.reshape(*stoch.shape[:-2], self.stoch_size * self.num_classes)
        
        # Then take mean across classes to get back to stoch_size dimension
        stoch = stoch.reshape(*stoch.shape[:-1], self.stoch_size, self.num_classes)
        stoch = stoch.mean(dim=-1)  # Average the one-hot encoding
        
        stoch = self.ln_stoch(stoch)
        
        # For mean/std, use average of categorical
        mean = dist.probs.mean(dim=-1)
        std = dist.probs.std(dim=-1)
        
        return RSSMState(mean, std, stoch, det)


class RSSMRepresentation(nn.Module):
    def __init__(
        self, transition_model: RSSMTransition, obs_encoder_feature_dim, action_size,
        stochastic_size=30, deterministic_size=200, hidden_size=200,
        activation=nn.ELU, unimix_ratio=0.01
    ):
        super().__init__()
        self.transition_model = transition_model
        self.obs_encoder_feature_dim = obs_encoder_feature_dim
        self.action_size = action_size
        self.stoch_size = stochastic_size
        self.det_size = deterministic_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.unimix_ratio = unimix_ratio

        self.ln_stoch = nn.LayerNorm(stochastic_size)
        
        self.num_classes = 32
        self.stoch_fc = nn.Sequential(
            nn.Linear(self.det_size + self.obs_encoder_feature_dim, self.hidden_size),
            self.activation(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.stoch_size * self.num_classes)
        )

    def initial_state(self, batch_size, device):
        """Initialize the RSSM state with zeros"""
        return RSSMState(
            torch.zeros(batch_size, self.stoch_size, device=device),
            torch.zeros(batch_size, self.stoch_size, device=device),
            torch.zeros(batch_size, self.stoch_size, device=device),
            torch.zeros(batch_size, self.det_size, device=device),
        )

    def forward(self, embed_obs, prev_action, prev_state: RSSMState):
        # Prior state
        prior_state = self.transition_model(prev_action, prev_state)
        
        x = torch.cat([prior_state.det, embed_obs], dim=-1)
        
        # Posterior with categorical distribution
        logits = self.stoch_fc(x)
        logits = logits.reshape(*logits.shape[:-1], self.stoch_size, self.num_classes)
        
        dist = OneHotDist(logits=logits, unimix_ratio=self.unimix_ratio)
        stoch = dist.sample()
        # Fix: same fix as in RSSMTransition
        stoch = stoch.reshape(*stoch.shape[:-2], self.stoch_size * self.num_classes)
        stoch = stoch.reshape(*stoch.shape[:-1], self.stoch_size, self.num_classes)
        stoch = stoch.mean(dim=-1)
        
        stoch = self.ln_stoch(stoch)
        
        mean = dist.probs.mean(dim=-1)
        std = dist.probs.std(dim=-1)

        posterior_state = RSSMState(mean, std, stoch, prior_state.det)

        return prior_state, posterior_state

class RSSMRollout(nn.Module):
    def __init__(
        self, representation_model: RSSMRepresentation,
        transition_model: RSSMTransition
    ):
        super().__init__()
        self.representation_model = representation_model
        self.transition_model = transition_model

    def forward(self, seq_len, embed_obses, actions, prev_state: RSSMState):
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
        imagined_priors = []
        prev_state = initial_state
        
        for t in range(horizon_length):
            action = actor(prev_state.stoch, prev_state.det)  
            next_state = self.transition_model(action, prev_state)
            imagined_priors.append(next_state)
            prev_state = next_state
        
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
        self.actions_size = np.prod(actions_size) if isinstance(actions_size, (list, tuple)) else actions_size
        actions_size = self.actions_size
        self.stoch_size = stochastic_size
        self.det_size = deterministic_size
        self.latent_size = stochastic_size + deterministic_size
        self.num_layers = num_layers
        self.dtype = dtype
        self.device = device

        self.observation_encoder = ObservationEncoder(
            shape=obs_shape, num_layers=num_layers, obs_encoder_feature_dim=obs_encoder_feature_dim,
            depth=num_filters
        )

        self.transition = RSSMTransition(
            actions_size, stochastic_size, deterministic_size, hidden_size
        )
        self.representation = RSSMRepresentation(
            self.transition, obs_encoder_feature_dim, actions_size,
            stochastic_size, deterministic_size, hidden_size
        )
        self.rollout = RSSMRollout(self.representation, self.transition)
        self.ln = nn.LayerNorm(self.latent_size)

    def forward(self, obs, prev_action, prev_state):
        state = self.get_state_representation(obs, prev_action, prev_state)
        return state
    
    def get_state_representation(self, obs, prev_action, prev_state):
        embed_obses = self.observation_encoder(obs)

        if prev_action is None:
            prev_action = torch.zeros(obs.size(0), self.actions_size, device=self.device)
        if prev_state is None:
            prev_state = self.representation.initial_state(prev_action.size(0), device=self.device)

        _, state = self.representation(embed_obses, prev_action, prev_state)
        return state
    
    def encode_sequence(self, obes, actions):
        seq_len, batch_size, ch, h, w = obes.size()
       
        prev_actions = actions[:-1]
        prev_action = torch.zeros(batch_size, self.actions_size, device=self.device, dtype=actions.dtype).unsqueeze(0)
        prev_actions = torch.cat([prev_action, prev_actions], dim=0)
        
        prev_state = self.representation.initial_state(batch_size, device=self.device)
        
        embed_obses = self.observation_encoder(obes)

        prior, posterior = self.rollout.rollout_representation(
            seq_len, embed_obses, prev_actions, prev_state
        )

        return prior, posterior


class RewardModel(nn.Module):
    def __init__(self, config, use_twohot=True):
        super().__init__()
        self.config = config['parameters']['dreamer']
        self.stochastic_size = self.config['stochastic_size']
        self.deterministic_size = self.config['deterministic_size']
        self.use_twohot = use_twohot
 
        if use_twohot:
            # Output 255 bins for two-hot encoding
            self.network = nn.Sequential(
                nn.Linear(self.stochastic_size + self.deterministic_size, 400),
                nn.ELU(),
                nn.Linear(400, 255)
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(self.stochastic_size + self.deterministic_size, 400),
                nn.ELU(),
                nn.Linear(400, 1)
            )
        
        # Initialize output layer to zeros
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, posterior, deterministic):
        seq_len, batch_size = posterior.shape[:2]
        x = torch.cat((posterior, deterministic), dim=-1)
        x = x.reshape(seq_len * batch_size, -1)
        x = self.network(x)
        x = x.reshape(seq_len, batch_size, -1)
        
        if self.use_twohot:
            dist = TwoHotDist(x, device=posterior.device)
        else:
            dist = td.Independent(
                td.Normal(x, torch.ones_like(x)), 
                1
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
        seq_len, batch_size = posterior.shape[:2]
        x = torch.cat((posterior, deterministic), dim=-1)
        x = x.reshape(seq_len * batch_size, -1)
        x = self.network(x)
        x = x.reshape(seq_len, batch_size, 1)
        dist = td.Bernoulli(logits=x.squeeze(-1))
        return dist