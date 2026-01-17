import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


class TwoHotDist:
    """Two-hot encoding distribution for critic value predictions"""
    def __init__(self, logits, low=-20.0, high=20.0, device='cuda'):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255, device=device)
        
    def mean(self):
        """Expected value using symexp transformation"""
        _mean = self.probs * self.buckets
        result = torch.sum(_mean, dim=-1, keepdim=True)
        # Apply symexp transformation
        return torch.sign(result) * (torch.exp(torch.abs(result)) - 1.0)
    
    def mode(self):
        return self.mean()

    def log_prob(self, x):
        """Compute log probability using two-hot encoding"""
        # Apply symlog transformation to target
        x = torch.sign(x) * torch.log(torch.abs(x) + 1.0)
        
        # Find closest buckets
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        
        # Clip to valid range
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        # Compute weights for two-hot encoding
        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        
        # Create two-hot target
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        
        # Compute log probabilities
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)
        
        return (target * log_pred).sum(-1)


class Critic(nn.Module):
    def __init__(
        self,
        stochastic_size=30,
        deterministic_size=200,
        hidden_size=400,
        num_layers=4,
        activation=nn.ELU,
        use_twohot=True,
    ):
        super().__init__()
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.use_twohot = use_twohot
        
        # Output 255 bins for two-hot encoding, or 1 for normal distribution
        output_size = 255 if use_twohot else 1
        
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
        
        # Initialize output layer to zeros (DreamerV3 technique)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, posterior, deterministic):
        """
        Args:
            posterior: (..., stochastic_size)
            deterministic: (..., deterministic_size)
        Returns:
            dist: Distribution over value predictions (TwoHot or Normal)
        """
        # Concatenate along last dimension
        x = torch.cat((posterior, deterministic), dim=-1)
        
        # Pass through network (operates on last dimension only)
        x = self.network(x)
        
        if self.use_twohot:
            # Create two-hot distribution
            dist = TwoHotDist(x, device=posterior.device)
        else:
            # Create Normal distribution with fixed std=1
            dist = Independent(Normal(x, torch.ones_like(x)), 1)
        
        return dist