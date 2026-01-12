import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

class DRIBO(nn.Module):
    def __init__(
        self, obses_shape, obs_encoder_feature_dim,
        encoder, encoder_target, device, output_type="continuous"
    ):
        super().__init__()
        self.device = device
        self.encoder = encoder
        self.encoder_target = encoder_target

        self.W = nn.Parameter(torch.rand(obs_encoder_feature_dim, obs_encoder_feature_dim))

    def encode(self, obses, actions, ema=False):
        seq_len = obses.size(1)
        
        if ema:
            with torch.no_grad():
                prior, post = self.encoder_target.encode_sequence(obses, actions)
        else:
            prior, post = self.encoder.encode_sequence(obses, actions)

        return prior, post
    
    def compute_logits(self, z1, z2):
        Wz = torch.matmul(self.W, z2.T)  # (z_dim, B)
        logits = torch.matmul(z1, Wz)  # (B, B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits
    
    def compute_skl(self, z1_dist, z2_dist):
        kl_1_2 = torch.mean(torch.distributions.kl.kl_divergence(z1_dist, z2_dist))
        kl_2_1 = torch.mean(torch.distributions.kl.kl_divergence(z2_dist, z1_dist))
        skl = (kl_1_2 + kl_2_1) / 2.
        return skl
    
    def compute_kl_balancing(self, z1_prior, z1_post):
        def get_dist(rssm_state, ema=False):
            if not ema:
                return td.independent.Independent(td.Normal(rssm_state.mean, rssm_state.std), 1)
            else:
                return td.independent.Independent(td.Normal(rssm_state.mean.detach(), rssm_state.std.detach()), 1)
        
        kl_t = 0.8 * torch.mean(torch.distributions.kl.kl_divergence(get_dist(z1_post, ema=True), get_dist(z1_prior)))
        kl_q = 0.2 * torch.mean(torch.distributions.kl.kl_divergence(get_dist(z1_post), get_dist(z1_prior, ema=True)))
        return kl_t + kl_q