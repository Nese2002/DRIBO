import torch
import numpy as np
import os
from collections import deque

def random_crop_gpu(imgs, out=84):
    """GPU-accelerated random crop using advanced indexing"""
    n, c, h, w = imgs.shape
    crop_max = h - out
    
    # Random crop coordinates
    w1 = torch.randint(0, crop_max + 1, (n,), device=imgs.device)
    h1 = torch.randint(0, crop_max + 1, (n,), device=imgs.device)
    
    # Create index grids (more efficient than loop)
    crop_h = torch.arange(out, device=imgs.device).view(1, out, 1).expand(n, out, out)
    crop_w = torch.arange(out, device=imgs.device).view(1, 1, out).expand(n, out, out)
    
    # Add offsets
    crop_h = crop_h + h1.view(n, 1, 1)
    crop_w = crop_w + w1.view(n, 1, 1)
    
    # Advanced indexing for batch crop
    batch_idx = torch.arange(n, device=imgs.device).view(n, 1, 1).expand(n, out, out)
    cropped = imgs[batch_idx, :, crop_h, crop_w]
    
    return cropped


class ReplayBuffer:
    """
    GPU-based replay buffer that stores everything on GPU to eliminate transfers.
    
    Tradeoffs:
    - Pro: Zero CPU-GPU transfer during sampling (much faster)
    - Pro: All operations stay on GPU (cropping, normalization)
    - Con: Uses GPU memory (limited vs CPU RAM)
    - Con: Slightly slower adds (CPU → GPU transfer once per transition)
    """
    def __init__(
        self,
        obs_shape,
        action_shape,
        capacity,
        batch_size,
        device,
        episode_len=None,
        image_size=84,
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.episode_len = episode_len
        
        # ✅ Store directly on GPU
        self.obses = torch.empty((capacity, *obs_shape), dtype=torch.uint8, device=device)
        self.actions = torch.empty((capacity, *action_shape), dtype=torch.float32, device=device)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self.not_dones = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        
        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        """Add single transition - converts numpy to torch and transfers to GPU"""
        # Transfer happens once here (acceptable cost)
        self.obses[self.idx] = torch.as_tensor(obs, device=self.device)
        self.actions[self.idx] = torch.as_tensor(action, device=self.device)
        self.rewards[self.idx] = reward
        self.not_dones[self.idx] = not done
        
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
    
    def add_batch(self, obs, action, reward, next_obs, done):
        """
        More efficient batch add - use this if you collect multiple transitions
        before adding to buffer
        """
        batch_size = obs.shape[0]
        idxs = torch.arange(self.idx, self.idx + batch_size) % self.capacity
        
        self.obses[idxs] = torch.as_tensor(obs, device=self.device)
        self.actions[idxs] = torch.as_tensor(action, device=self.device)
        self.rewards[idxs] = torch.as_tensor(reward, device=self.device).unsqueeze(-1)
        self.not_dones[idxs] = torch.as_tensor(~done, device=self.device).unsqueeze(-1)
        
        self.idx = (self.idx + batch_size) % self.capacity
        self.full = self.full or self.idx < batch_size

    def sample_sequential_idxs(self, batch_size, seq_len):
        """Sample indices for sequential data"""
        last_idx = self.capacity - seq_len if self.full else self.idx - seq_len
        
        # ✅ Generate indices on GPU
        idx = torch.randint(0, last_idx, (batch_size,), device=self.device)
        
        if self.episode_len is not None:
            pos_in_episode = idx % self.episode_len
            mask = pos_in_episode > self.episode_len - seq_len
            idx[mask] = (idx[mask] // self.episode_len) * self.episode_len + seq_len
        
        # Create sequence indices
        idxs = idx.unsqueeze(1) + torch.arange(seq_len, device=self.device).unsqueeze(0)
        idxs = idxs.t().reshape(-1)  # Shape: (seq_len * batch_size,)
        
        return idxs

    def sample_multi_view(self, batch_size, seq_len):
        """
        Sample sequences with data augmentation.
        ✅ ZERO CPU-GPU TRANSFERS - everything stays on GPU!
        """
        idxs = self.sample_sequential_idxs(batch_size, seq_len)
        
        # ✅ Index directly on GPU (no transfer!)
        obses = self.obses[idxs]
        positives = self.obses[idxs]  # Same data, different augmentation
        
        # ✅ Crop on GPU (parallel)
        obses = random_crop_gpu(obses, out=self.image_size)
        positives = random_crop_gpu(positives, out=self.image_size)
        
        # ✅ Normalize on GPU
        obses = obses.float()
        positives = positives.float()
        
        # Reshape to (seq_len, batch_size, ...)
        obses = obses.reshape(seq_len, batch_size, *obses.shape[-3:])
        positives = positives.reshape(seq_len, batch_size, *positives.shape[-3:])
        
        # ✅ Get other data (already on GPU)
        actions = self.actions[idxs].reshape(seq_len, batch_size, -1)
        rewards = self.rewards[idxs].reshape(seq_len, batch_size, -1)
        not_dones = self.not_dones[idxs].reshape(seq_len, batch_size, -1)

        return obses, positives, actions, rewards, not_dones
    
    def save(self, save_dir):
        """Save buffer to disk"""
        if self.idx == self.last_save:
            return
        
        path = os.path.join(save_dir, f'{self.last_save}_{self.idx}.pt')
        
        # Transfer to CPU for saving
        payload = {
            'obses': self.obses[self.last_save:self.idx].cpu(),
            'actions': self.actions[self.last_save:self.idx].cpu(),
            'rewards': self.rewards[self.last_save:self.idx].cpu(),
            'not_dones': self.not_dones[self.last_save:self.idx].cpu(),
        }
        
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        """Load buffer from disk"""
        chunks = sorted(os.listdir(save_dir), key=lambda x: int(x.split('_')[0]))
        
        for chunk in chunks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path, map_location=self.device)
            
            assert self.idx == start
            self.obses[start:end] = payload['obses']
            self.actions[start:end] = payload['actions']
            self.rewards[start:end] = payload['rewards']
            self.not_dones[start:end] = payload['not_dones']
            self.idx = end

    def __len__(self):
        return self.capacity if self.full else self.idx