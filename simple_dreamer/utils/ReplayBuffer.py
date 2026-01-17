import torch
import numpy as np
import os
from collections import deque, namedtuple
from torch.utils.data import Dataset

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
        obs_shape=(3,100,100),
        action_shape=6,
        capacity=50000,
        batch_size=8,
        device=None,
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

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype) #[capacity, T, B, C, H, W]
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False
        self.episode_len = episode_len

    def add(self, obs, action, reward, next_obs, done):

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

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
        obses = self.obses[idxs] # (batch_size * seq_len, C, H, W) tensor of observations
        positives = obses.copy()

        obses = random_crop(obses, out=self.image_size) # augmented first view
        positives = random_crop(positives, out=self.image_size) #augmented second view

        obses = torch.as_tensor(obses, device=self.device).float().reshape(seq_len, batch_size, *obses.shape[-3:]) #(seq_len, batch_size, C, H, W)
        positives = torch.as_tensor(positives, device=self.device).float().reshape(seq_len, batch_size, *obses.shape[-3:])
        actions = torch.as_tensor(self.actions[idxs], device=self.device).reshape(seq_len, batch_size, -1)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device).reshape(seq_len, batch_size)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device).reshape(seq_len, batch_size)

        return obses, positives, actions, rewards, not_dones
    
    def sample(self, batch_size, chunk_size):
        obses, positives, actions, rewards, not_dones = self.sample_multi_view(batch_size, chunk_size)
        
        # Use first view only (or keep augmentation if you want)
        return {
            'observations': obses,
            'positives': positives,
            'actions': actions,
            'rewards': rewards,
            'next_observations': obses[1:],  # shift by 1 timestep
            'done': 1 - not_dones  # ReplayBuffer stores not_dones, Dreamer uses done
        }
    
    def save(self, save_dir):
        """Save complete replay buffer state"""
        path = os.path.join(save_dir, 'replay_buffer.npz')
        
        size = self.capacity if self.full else self.idx
        
        np.savez_compressed(
            path,
            obses=self.obses[:size],
            next_obses=self.next_obses[:size],
            actions=self.actions[:size],
            rewards=self.rewards[:size],
            not_dones=self.not_dones[:size],
            idx=self.idx,
            full=self.full,
            last_save=self.last_save
        )

    def load(self, load_dir):
        """Load complete replay buffer state"""
        path = os.path.join(load_dir, 'replay_buffer.npz')
        
        if not os.path.exists(path):
            print(f"No replay buffer found at {path}")
            return
        
        data = np.load(path)
        
        # Restore data
        size = len(data['obses'])
        self.obses[:size] = data['obses']
        self.next_obses[:size] = data['next_obses']
        self.actions[:size] = data['actions']
        self.rewards[:size] = data['rewards']
        self.not_dones[:size] = data['not_dones']
        
        # Restore metadata
        self.idx = int(data['idx'])
        self.full = bool(data['full'])
        self.last_save = int(data['last_save'])

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
        return self.capacity if self.full else self.idx