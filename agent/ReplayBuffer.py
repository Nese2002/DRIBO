import torch
import numpy as np
import os
from collections import deque, namedtuple
from torch.utils.data import Dataset

def random_crop_gpu(imgs, out=84):
    """GPU-accelerated random crop"""
    # imgs: torch tensor on GPU (N, C, H, W)
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

def random_crop(imgs, out=84):

    n, c, h, w = imgs.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = np.empty((n, c, out, out), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
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
        obs_dtype = np.uint8

        # ✅ Store on GPU directly (as uint8 to save memory)
        self.obses = torch.empty((capacity, *obs_shape), dtype=torch.uint8, device=device)
        self.actions = torch.empty((capacity, *action_shape), dtype=torch.float32, device=device)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self.not_dones = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        

        self.idx = 0
        self.last_save = 0
        self.full = False
        self.episode_len = episode_len

    def add(self, obs, action, reward, next_obs, done):

        self.obses[self.idx].copy_(torch.from_numpy(obs))
        self.actions[self.idx].copy_(torch.from_numpy(action))
        self.rewards[self.idx] = reward
        self.not_dones[self.idx] = not done
        
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    
    def sample_sequential_idxs(self, batch_size, seq_len):
        # Randomly samples batch_size starting indices
        last_idx = self.capacity - seq_len if self.full else self.idx - seq_len
        idx = np.random.randint(0, last_idx, size=batch_size)
        pos_in_episode = idx % self.episode_len

        # Move index that would cross episode boundaries to a safe start inside the SAME trajectory
        idx[pos_in_episode > self.episode_len - seq_len] = idx[pos_in_episode > self.episode_len - seq_len] // self.episode_len * self.episode_len + seq_len
        
        # Build sequential indices
        idxs = np.zeros((batch_size, seq_len), dtype=int)    #Each row corresponds to one sampled sequence
                                                                #Each column corresponds to a time step inside that sequence
        
        # Fill sequences (idxs[i] is the i-th sequence)
        for i in range(batch_size):
            idxs[i] = np.arange(idx[i], idx[i] + seq_len)
        return idxs.transpose().reshape(-1)

    def sample_multi_view(self, batch_size, seq_len):
        idxs = self.sample_sequential_idxs(batch_size, seq_len)
        
        # ✅ Already on GPU!
        obses = self.obses[idxs]
        positives = self.obses[idxs]
        
        # ✅ Crop on GPU
        obses = random_crop_gpu(obses, out=self.image_size).float()
        positives = random_crop_gpu(positives, out=self.image_size).float()
        
        # Reshape
        obses = obses.reshape(seq_len, batch_size, *obses.shape[-3:])
        positives = positives.reshape(seq_len, batch_size, *positives.shape[-3:])
        
        actions = self.actions[idxs].reshape(seq_len, batch_size, -1)
        rewards = self.rewards[idxs].reshape(seq_len, batch_size)
        not_dones = self.not_dones[idxs].reshape(seq_len, batch_size)

        return obses, positives, actions, rewards, not_dones
    
    
    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        episode = os.episode.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, episode)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            episode = os.episode.join(save_dir, chunk)
            payload = torch.load(episode)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
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