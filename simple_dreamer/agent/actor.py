import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, TransformedDistribution, OneHotCategorical
from torch.distributions.transforms import TanhTransform

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


class Actor(nn.Module):
    def __init__(
        self, discrete_action_bool, action_size, 
        stochastic_size=30, deterministic_size=200,
        hidden_size=400, num_layers=4, activation=nn.ELU,
        mean_scale=5, init_std=0, min_std=1e-4
    ):
        super().__init__()
        self.discrete_action_bool = discrete_action_bool
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.mean_scale = mean_scale
        self.init_std = init_std
        self.min_std = min_std
        
        # Output size: action_size for discrete, 2*action_size for continuous (mean+std)
        output_size = 2 * action_size
        
        # Build network manually (replacing build_network)
        layers = []
        layers.append(nn.Linear(stochastic_size + deterministic_size, hidden_size))
        layers.append(activation())
        
        for i in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation())
        
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)
        self.network.apply(initialize_weights)

    def forward(self, stochastic, deterministic):
        """
        Args:
            stochastic: (..., stochastic_size)
            deterministic: (..., deterministic_size)
        Returns:
            action: (..., action_size)
        """
        # Concatenate along last dimension
        x = torch.cat((stochastic, deterministic), dim=-1)
        
        # Pass through network (operates on last dimension only)
        x = self.network(x)
        
        if self.discrete_action_bool:
            # Discrete actions
            dist = OneHotCategorical(logits=x)
            action = dist.sample() + dist.probs - dist.probs.detach()
        else:
            # Continuous actions (replacing create_normal_dist)
            mean, std = torch.chunk(x, 2, dim=-1)
            
            # Apply activation and scaling to mean
            mean = mean / self.mean_scale
            mean = torch.tanh(mean)
            mean = self.mean_scale * mean
            
            # Process std
            std = F.softplus(std + self.init_std) + self.min_std
            
            # Create distribution
            dist = Normal(mean, std)
            dist = TransformedDistribution(dist, TanhTransform())
            dist = Independent(dist, 1)
            
            # Sample action
            action = dist.rsample()
        
        return action