import torch
import numpy as np
import os
from torch.utils.data import Dataset

def random_crop(imgs, out=84):
    """Vectorized random crop - no loops!"""
    n, c, h, w = imgs.shape
    crop_max = h - out + 1
    
    # Generate all random offsets at once
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    
    # Create index grids for cropping
    # Shape: (n, c, out, out)
    n_idx = np.arange(n)[:, None, None, None]
    c_idx = np.arange(c)[None, :, None, None]
    h_idx = (np.arange(out)[None, None, :, None] + h1[:, None, None, None])
    w_idx = (np.arange(out)[None, None, None, :] + w1[:, None, None, None])
    
    # Vectorized indexing - single operation!
    return imgs[n_idx, c_idx, h_idx, w_idx]


class ReplayBuffer(Dataset):
    def __init__(
        self,
        obs_shape,
        action_shape,
        capacity,
        batch_size,
        device,
        episode_len=None,
        image_size=84,
        transform=None
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        self.episode_len = episode_len
        
        obs_dtype = np.uint8

        # Pre-allocate arrays
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        """Add transition to buffer"""
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    
    def sample_sequential_idxs(self, batch_size, seq_len):
        """Fully vectorized sequential sampling - no loops!"""
        # Determine valid range
        last_idx = self.capacity - seq_len if self.full else self.idx - seq_len
        
        # Sample starting indices
        idx = np.random.randint(0, last_idx, size=batch_size)
        
        # Vectorized episode boundary handling
        pos_in_episode = idx % self.episode_len
        crosses_boundary = pos_in_episode > self.episode_len - seq_len
        
        # Adjust indices that cross boundaries (vectorized)
        idx = np.where(
            crosses_boundary,
            (idx // self.episode_len) * self.episode_len + seq_len,
            idx
        )
        
        # Build sequential indices using broadcasting
        # idx[:, None] has shape (batch_size, 1)
        # np.arange(seq_len) has shape (seq_len,)
        # Result has shape (batch_size, seq_len)
        idxs = idx[:, None] + np.arange(seq_len)[None, :]
        
        # Transpose and flatten: (seq_len, batch_size) -> (seq_len * batch_size,)
        return idxs.T.ravel()

    def sample_multi_view(self, batch_size, seq_len):
        """Sample sequences with data augmentation"""
        # Get sequential indices (vectorized)
        idxs = self.sample_sequential_idxs(batch_size, seq_len)
        
        # Index into arrays (vectorized)
        obses = self.obses[idxs]  # (batch_size * seq_len, C, H, W)
        
        # Create copy for second view
        positives = obses.copy()

        # Apply random crops (vectorized)
        obses = random_crop(obses, out=self.image_size)
        positives = random_crop(positives, out=self.image_size)

        # Reshape to (seq_len, batch_size, ...)
        target_shape = (seq_len, batch_size, *obses.shape[-3:])
        
        obses = torch.as_tensor(obses, device=self.device).float().reshape(target_shape)
        positives = torch.as_tensor(positives, device=self.device).float().reshape(target_shape)
        
        # Process other arrays (vectorized indexing + reshape)
        actions = torch.as_tensor(self.actions[idxs], device=self.device).reshape(seq_len, batch_size, -1)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device).reshape(seq_len, batch_size, 1).squeeze(-1)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device).reshape(seq_len, batch_size, 1).squeeze(-1)

        return obses, positives, actions, rewards, not_dones
    
    
    def save(self, save_dir):
        """Save buffer to disk"""
        if self.idx == self.last_save:
            return
        
        # Fixed: os.path.join instead of os.episode.join
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        """Load buffer from disk"""
        chunks = os.listdir(save_dir)
        chunks = sorted(chunks, key=lambda x: int(x.split('_')[0]))  # Fixed: 'chucks' typo
        
        for chunk in chunks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)  # Fixed: os.path.join
            payload = torch.load(path)
            
            assert self.idx == start
            
            # Vectorized assignment
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            
            self.idx = end

    def __getitem__(self, idx):
        """Get single item (for Dataset interface)"""
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )[0]
        
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity