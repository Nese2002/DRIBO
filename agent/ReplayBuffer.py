import torch
import numpy as np
import os
from collections import deque, namedtuple
from torch.utils.data import Dataset

def random_crop_gpu(imgs, out=84):
    """GPU-accelerated random crop"""
    n, c, h, w = imgs.shape
    crop_max = h - out
    
    # Random crop coordinates
    w1 = torch.randint(0, crop_max + 1, (n,), device=imgs.device)
    h1 = torch.randint(0, crop_max + 1, (n,), device=imgs.device)
    
    # Crop all images in parallel on GPU
    cropped = torch.empty((n, c, out, out), dtype=imgs.dtype, device=imgs.device)
    for i in range(n):
        cropped[i] = imgs[i, :, h1[i]:h1[i] + out, w1[i]:w1[i] + out]
    
    return cropped


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
        
        # ✅ KEEP ON CPU - numpy is fast for single adds
        obs_dtype = np.uint8
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        
        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        # ✅ FAST: Direct numpy copy on CPU
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)
        
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
    
    def sample_sequential_idxs(self, batch_size, seq_len):
        last_idx = self.capacity - seq_len if self.full else self.idx - seq_len
        idx = np.random.randint(0, last_idx, size=batch_size)
        pos_in_episode = idx % self.episode_len
        
        idx[pos_in_episode > self.episode_len - seq_len] = \
            idx[pos_in_episode > self.episode_len - seq_len] // self.episode_len * self.episode_len + seq_len
        
        idxs = np.zeros((batch_size, seq_len), dtype=int)
        for i in range(batch_size):
            idxs[i] = np.arange(idx[i], idx[i] + seq_len)
        return idxs.transpose().reshape(-1)

    def sample_multi_view(self, batch_size, seq_len):
        idxs = self.sample_sequential_idxs(batch_size, seq_len)
        
        # ✅ Get from CPU numpy (fast indexing)
        obses_np = self.obses[idxs]
        
        # ✅ Transfer to GPU once (efficient batch transfer)
        obses = torch.as_tensor(obses_np, device=self.device)
        positives = torch.as_tensor(obses_np, device=self.device)  # Share the same data
        
        # ✅ Crop on GPU (parallel)
        obses = random_crop_gpu(obses, out=self.image_size).float()
        positives = random_crop_gpu(positives, out=self.image_size).float()
        
        # Reshape
        obses = obses.reshape(seq_len, batch_size, *obses.shape[-3:])
        positives = positives.reshape(seq_len, batch_size, *positives.shape[-3:])
        
        # Transfer other data
        actions = torch.as_tensor(self.actions[idxs], device=self.device).reshape(seq_len, batch_size, -1)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device).reshape(seq_len, batch_size)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device).reshape(seq_len, batch_size)

        return obses, positives, actions, rewards, not_dones
    
    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chunks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chunks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.actions[start:end] = payload[1]
            self.rewards[start:end] = payload[2]
            self.not_dones[start:end] = payload[3]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(0, self.capacity if self.full else self.idx, size=1)[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)

        return obs, action, reward, not_done

    def __len__(self):
        return self.capacity