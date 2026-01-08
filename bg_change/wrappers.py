import gymnasium as gym
from gymnasium import spaces
import glob
import os
from dm_control import suite
from dm_env import specs
import numpy as np
from collections import deque

import bg_change.bg_change as bg_change


def _spec_to_box(spec):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(np.float32)
    high = np.concatenate(maxs, axis=0).astype(np.float32)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)

class DMCWrapper(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        domain_name,
        task_name,
        resource_files,
        img_source,
        total_frames,
        task_kwargs=None,
        visualize_reward={},
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        frame_stack=1,
        environment_kwargs=None,
        render_mode=None,
    ):
        assert 'random' in task_kwargs, \
            'please specify a seed, for deterministic behaviour'
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._img_source = img_source
        self.render_mode = render_mode

        self._frame_stack = frame_stack
        self._frames = deque([], maxlen=frame_stack)

        # create control suite environment
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            environment_kwargs=environment_kwargs
        )

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()])
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        # create observation space
        if from_pixels:
            self._observation_space = spaces.Box(
                low=0, high=255,
                shape=[3*frame_stack, height, width], dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values()
            )

        self._internal_state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self._env.physics.get_state().shape,
            dtype=np.float32
        )

        # background
        if img_source is not None:
            shape2d = (height, width)
            
            files = glob.glob(os.path.expanduser(resource_files))
            assert len(
                files
            ), "Pattern {} does not match any files".format(
                resource_files
            )
            print(len(files))
    
            self._bg_source = bg_change.RandomVideoSource(
                shape2d, files,
                total_frames=total_frames
            )
           
        # set seed
        self._seed = task_kwargs.get('random', 1)
        self.seed(seed=self._seed)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_noisy_obs(self, time_step):
        if self._from_pixels:
            obs = self._render_frame(
                height=self._height,
                width=self._width,
                camera_id=self._camera_id
            )
            if self._img_source is not None:
                mask = np.logical_and(
                    (obs[:, :, 2] > obs[:, :, 1]),
                    (obs[:, :, 2] > obs[:, :, 0])
                )  # hardcoded for dmc
                bg = self._bg_source.get_image()
                obs[mask] = bg[mask]
            obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    def _get_obs(self):
        assert len(self._frames) == self._frame_stack
        return np.concatenate(list(self._frames), axis=0)

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def internal_state_space(self):
        return self._internal_state_space

    @property
    def action_space(self):
        return self._norm_action_space

    def seed(self, seed=None):
        """
        Gymnasium seed() returns a tuple of seeds.
        """
        if seed is not None:
            self._seed = seed
        self._true_action_space.seed(self._seed)
        self._norm_action_space.seed(self._seed)
        self._observation_space.seed(self._seed)
        return (self._seed,)

    def step(self, action):
        """
        Gymnasium step() returns 5 values: obs, reward, terminated, truncated, info
        - terminated: episode ended due to task completion/failure
        - truncated: episode ended due to time limit
        """
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        info = {'internal_state': self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        
        obs = self._get_noisy_obs(time_step)
        self._frames.append(obs)
        info['discount'] = time_step.discount
        
        # In DMC, episodes end naturally (terminated=True, truncated=False)
        # If you want to handle time limits separately, you'd need to track steps
        terminated = done
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Gymnasium reset() returns (obs, info) tuple.
        seed parameter allows resetting with a new seed.
        """
        if seed is not None:
            self.seed(seed)
            
        time_step = self._env.reset()
        obs = self._get_noisy_obs(time_step)
        for _ in range(self._frame_stack):
            self._frames.append(obs)
        
        info = {}
        return self._get_obs(), info

    def _render_frame(self, height=None, width=None, camera_id=0):
        """Internal rendering method."""
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )

    def render(self):
        """
        Gymnasium render() API:
        - No mode parameter (set via render_mode in __init__)
        - Returns the rendered frame for rgb_array mode
        - Returns None for human mode (displays directly)
        """
        if self.render_mode == "rgb_array":
            obs = self._render_frame(
                height=self._height,
                width=self._width,
                camera_id=self._camera_id
            )
            
            # Apply background substitution if enabled
            if self._from_pixels and self._img_source is not None:
                mask = np.logical_and(
                    (obs[:, :, 2] > obs[:, :, 1]),
                    (obs[:, :, 2] > obs[:, :, 0])
                )
                bg = self._bg_source.get_image()
                obs[mask] = bg[mask]
            
            return obs
        return None

    def close(self):
        """Clean up resources."""
        if hasattr(self, '_env'):
            self._env.close()