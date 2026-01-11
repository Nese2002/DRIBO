import torch
import numpy as np
import gym
import os
import sys
from collections import deque, namedtuple
import random
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
        obs_shape,
        action_shape,
        capacity,
        batch_size,
        device,
        path_len=None,
        image_size=84,
        transform=None
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        obs_dtype = np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False
        self._path_len = path_len

    def add(self, obs, action, reward, next_obs, done):

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    
    def _sample_sequential_idx(self, batch_size, seq_len):
        idx = np.random.randint(
            0, self.capacity - seq_len if self.full else self.idx - seq_len, size=batch_size
        )
        pos_in_path = idx - idx // self._path_len * self._path_len
        idx[pos_in_path > self._path_len - seq_len] = idx[pos_in_path > self._path_len - seq_len] // self._path_len * self._path_len + seq_len
        idxs = np.zeros((batch_size, seq_len), dtype=np.int)
        for i in range(batch_size):
            idxs[i] = np.arange(idx[i], idx[i] + seq_len)
        return idxs.transpose().reshape(-1)

    def sample_multi_view(self, batch_size, seq_len):
        # start = time.time()
        idxs = self._sample_sequential_idx(batch_size, seq_len)
        obses = self.obses[idxs]
        pos = obses.copy()

        obses = random_crop(obses, out=self.image_size)
        pos = random_crop(pos, out=self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()\
            .reshape(seq_len, batch_size, *obses.shape[-3:])
        actions = torch.as_tensor(self.actions[idxs], device=self.device)\
            .reshape(seq_len, batch_size, -1)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)\
            .reshape(seq_len, batch_size)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)\
            .reshape(seq_len, batch_size)

        pos = torch.as_tensor(pos, device=self.device).float()\
            .reshape(seq_len, batch_size, *obses.shape[-3:])
        mib_kwargs = dict(view1=obses, view2=pos,
                          time_anchor=None, time_pos=None)

        return obses, actions, rewards, not_dones, mib_kwargs
    
    
    def save(self, save_dir):
        if self.idx == self.last_save:
            return
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
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
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