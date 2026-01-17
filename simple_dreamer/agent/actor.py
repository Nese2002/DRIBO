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
        self.action_size = action_size
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.mean_scale = mean_scale
        self.init_std = init_std
        self.min_std = min_std
        
        # Output size: action_size for discrete, 2*action_size for continuous (mean+std)
        output_size = action_size if discrete_action_bool else 2 * action_size
        
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

    def forward(self, stochastic, deterministic, deterministic_action=False):
        """
        Forward pass of the actor network.
        
        Args:
            stochastic: (..., stochastic_size) - Stochastic state from RSSM
            deterministic: (..., deterministic_size) - Deterministic state from RSSM
            deterministic_action: bool - If True, return mean/mode action (for evaluation)
                                        If False, sample from distribution (for training)
        
        Returns:
            action: (..., action_size) - Sampled or deterministic action
        """
        # Concatenate along last dimension
        x = torch.cat((stochastic, deterministic), dim=-1)
        
        # Pass through network
        x = self.network(x)
        
        if self.discrete_action_bool:
            # Discrete actions
            if deterministic_action:
                # Use argmax for deterministic evaluation
                action_probs = F.softmax(x, dim=-1)
                action_indices = torch.argmax(action_probs, dim=-1, keepdim=True)
                action = F.one_hot(action_indices.squeeze(-1), num_classes=self.action_size).float()
            else:
                # Sample from categorical distribution for training
                dist = OneHotCategorical(logits=x)
                action = dist.sample() + dist.probs - dist.probs.detach()
        else:
            # Continuous actions
            mean, std = torch.chunk(x, 2, dim=-1)
            
            # Apply activation and scaling to mean
            mean = mean / self.mean_scale
            mean = torch.tanh(mean)
            mean = self.mean_scale * mean
            
            if deterministic_action:
                # Return mean action for deterministic evaluation
                action = torch.tanh(mean)
            else:
                # Process std and sample for training
                std = F.softplus(std + self.init_std) + self.min_std
                
                # Create distribution
                dist = Normal(mean, std)
                dist = TransformedDistribution(dist, TanhTransform())
                dist = Independent(dist, 1)
                
                # Sample action
                action = dist.rsample()
        
        return action
    
    def get_action_dist(self, stochastic, deterministic):
        """
        Get the action distribution for computing entropy or log probabilities.
        Useful for advanced RL algorithms that need distribution info.
        
        Args:
            stochastic: (..., stochastic_size)
            deterministic: (..., deterministic_size)
        
        Returns:
            dist: Action distribution
        """
        x = torch.cat((stochastic, deterministic), dim=-1)
        x = self.network(x)
        
        if self.discrete_action_bool:
            dist = OneHotCategorical(logits=x)
        else:
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
        
        return dist