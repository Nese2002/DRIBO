import torch
import torch.nn as nn
import torch.distributions as td

def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

class Decoder(nn.Module):
    def __init__(
        self, observation_shape, stochastic_size=30, deterministic_size=200,
        depth=32, kernel_size=5, stride=2, activation=nn.ReLU
    ):
        super().__init__()
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.observation_shape = observation_shape
        
        self.network = nn.Sequential(
            # Start with 2x2 instead of 1x1 to reach 84x84
            nn.Linear(
                self.deterministic_size + self.stochastic_size, depth * 32 * 2 * 2
            ),
            nn.Unflatten(1, (depth * 32, 2, 2)),  # Changed: now 2x2 instead of 1x1
            nn.ConvTranspose2d(depth * 32, depth * 4, kernel_size, stride),  # 2->7
            activation(),
            nn.ConvTranspose2d(depth * 4, depth * 2, kernel_size + 1, stride),  # 7->18
            activation(),
            nn.ConvTranspose2d(depth * 2, depth * 1, kernel_size + 1, stride),  # 18->40
            activation(),
            nn.ConvTranspose2d(depth * 1, observation_shape[0], kernel_size + 1, stride),  # 40->84
        )
        self.network.apply(initialize_weights)

    def forward(self, posterior_stoch, posterior_det):
        """
        Args:
            posterior_stoch: (seq_len, batch_size, stoch_size)
            posterior_det: (seq_len, batch_size, det_size)
        Returns:
            dist: Independent Normal distribution over reconstructed observations
        """
        # Get seq_len and batch dimensions
        seq_len, batch_size = posterior_stoch.shape[:2]
        
        # Concatenate stochastic and deterministic parts
        x = torch.cat([posterior_stoch, posterior_det], dim=-1)
        
        # Flatten seq_len and batch dimensions
        x = x.reshape(seq_len * batch_size, -1)
        
        # Pass through decoder network
        x = self.network(x)
        
        # Reshape back to (seq_len, batch_size, C, H, W)
        x = x.reshape(seq_len, batch_size, *self.observation_shape)
        
        # Create Normal distribution
        dist = td.Independent(
            td.Normal(x, torch.ones_like(x)), 
            len(self.observation_shape)
        )
        
        return dist