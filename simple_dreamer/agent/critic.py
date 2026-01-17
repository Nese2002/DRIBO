import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


class Critic(nn.Module):
    def __init__(
        self,
        stochastic_size=30,
        deterministic_size=200,
        hidden_size=400,
        num_layers=4,
        activation=nn.ELU,
    ):
        super().__init__()
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        
        # Output size: 1 for value prediction
        output_size = 1
        
        # Build network manually
        layers = []
        layers.append(nn.Linear(stochastic_size + deterministic_size, hidden_size))
        layers.append(activation())
        
        for i in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation())
        
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)
        self.network.apply(initialize_weights)

    def forward(self, posterior, deterministic):
        """
        Args:
            posterior: (..., stochastic_size)
            deterministic: (..., deterministic_size)
        Returns:
            dist: Normal distribution over value predictions
        """
        # Concatenate along last dimension
        x = torch.cat((posterior, deterministic), dim=-1)
        
        # Pass through network (operates on last dimension only)
        x = self.network(x)
        
        # Create distribution with fixed std=1
        dist = Normal(x, torch.ones_like(x))
        dist = Independent(dist, 1)
        
        return dist