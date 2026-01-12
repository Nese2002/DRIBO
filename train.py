import gymnasium as gym
from bg_change import make
import numpy as np
import cv2
import torch
import argparse
import os
import time
from tqdm import tqdm

from utils.video_recorder import VideoRecorder
from agent.ReplayBuffer import ReplayBuffer
from agent.DRIBOSacAgent import DRIBOSacAgent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--render', default=None) # "rgb_array" to render
    args = parser.parse_args()
    return args

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False  


def main():
#     video_recorder = VideoRecorder(dir_name='./videos', height=480, width=640)
#     video_recorder.init(enabled=True)

#     env = make(
#     domain_name='cheetah',
#     task_name='run',
#     resource_files='dataset/train/*.avi',
#     total_frames=1000,
#     seed=42,
#     render_mode="rgb_array",   # 🔑 REQUIRED
# )
    
#     obs, info = env.reset(seed=42)

#     for step in range(300):
#         action = env.action_space.sample()

#         if step % 50 == 0:
#             print(f"Step {step}, Action: {action}, Obs mean: {obs.mean()}")
#         obs, reward, terminated, truncated, info = env.step(action)
#         video_recorder.record(env)
#         # frame = env.render()  # (H, W, 3), uint8
#         # cv2.imshow("DMC + Video Background", frame[:, :, ::-1])  # RGB → BGR

#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break

#         if terminated or truncated:
#             obs, info = env.reset()
    
#     video_recorder.save('episode_001.mp4')
#     env.close()
#     cv2.destroyAllWindows()
   
    args = parse_args()
    domain_name = args.domain_name
    task_name = args.task_name
    render = args.render
    frame_skip=1
    frame_stack=1
    img_size = 100
    augmented_img_size = 84
    replay_buffer_capacity = 100000
    resource_files = 'dataset/train/*.avi'
    eval_resource_files = 'dataset/test/*.avi'
    total_frames = 1000
    save_video = False

    batch_size = 128
    episode_len = total_frames // frame_skip
    num_train_steps = 880000
    eval_freq = 10000
    init_step = 1000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create environment 
    env = make(
        domain_name=domain_name,
        task_name=task_name,
        resource_files=resource_files,
        total_frames=total_frames,
        frame_skip=frame_skip,
        frame_stack=frame_stack,
        seed= 42,
        width= img_size,
        height= img_size,
        render_mode=render, 
        extra='train' 
    )

    eval_env = make(
        domain_name=domain_name,
        task_name=task_name,
        resource_files=resource_files,
        total_frames=total_frames,
        frame_skip=frame_skip,
        frame_stack=frame_stack,
        seed= 42,
        width= img_size,
        height= img_size,
        render_mode=render,   
        extra='eval',
    )

    # make directory
    env_name = domain_name + '-' + task_name
    work_dir = "./log" + '/' + env_name
    os.makedirs(work_dir, exist_ok=True)  
    video_dir = os.path.join(work_dir, 'video')
    os.makedirs(video_dir, exist_ok=True)
    model_dir = os.path.join(work_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    buffer_dir = os.path.join(work_dir, 'buffer')
    os.makedirs(buffer_dir, exist_ok=True)
    
    if(save_video):
        video = VideoRecorder(video_dir)

    # Shape definition
    action_shape = env.action_space.shape
    obs_shape = (3*frame_stack, img_size, img_size)
    augmented_obs_shape = (3*frame_stack, augmented_img_size, augmented_img_size)

    # Replay buffer
    replay_buffer = ReplayBuffer(
        obs_shape,
        action_shape,
        capacity = replay_buffer_capacity,
        batch_size = batch_size, 
        episode_len = episode_len,
        device= device,
        image_size=augmented_img_size
    )

    # Agent
    agent = DRIBOSacAgent(
        obses_shape= augmented_obs_shape,
        actions_shape= action_shape,
        device= device
    )

    episode, episode_reward, terminated = 0, 0, True
    max_mean_ep_reward = 0

    pbar = tqdm(range(num_train_steps), desc="Training")

    from torch.profiler import profile, ProfilerActivity
    
    profiler = None
    profile_start = init_step + 100  # Start profiling after init_step
    profile_end = profile_start + 10  # Profile 10 steps

    for t in pbar:
        # if t %  eval_freq == 0:
        #     mean_ep_reward = evaluate(eval_env, agent, video, args.num_eval_episodes, t)
        #     if mean_ep_reward > max_mean_ep_reward:
        #         max_mean_ep_reward = mean_ep_reward
        #         agent.save_DRIBO(model_dir, t)
        #         replay_buffer.save(buffer_dir)

        if t == profile_start:
            print(f"\n🔍 Starting profiler at step {t}")
            profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True
            )
            profiler.__enter__()

        if terminated:
            obs,_ = env.reset()
            prev_state = None
            prev_action = None
            terminated = False
            episode_reward = 0
            episode_step = 0
            episode += 1

        # Random exploration phase
        if t < init_step:
            action = env.action_space.sample()

        # Policy-based action selection
        else:
            with eval_mode(agent.encoder, agent.actor):
                action, prev_action, prev_state = agent.sample_action(obs, prev_action, prev_state)

        # Training updates
        if t >= init_step:
            agent.update(replay_buffer, t)

        next_obs, reward, terminated, truncated, info = env.step(action)

        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(terminated)
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1

        # Stop profiler and print results
        if t == profile_end and profiler is not None:
            profiler.__exit__(None, None, None)
            print("\n" + "="*80)
            print("PROFILING RESULTS")
            print("="*80)
            print(profiler.key_averages().table(
                sort_by="cuda_time_total", 
                row_limit=20
            ))
            
            # Save detailed trace for Chrome
            trace_path = os.path.join(work_dir, 'profile_trace.json')
            profiler.export_chrome_trace(trace_path)
            print(f"\n💾 Detailed trace saved to: {trace_path}")
            print("   View in Chrome at: chrome://tracing")
            print("="*80 + "\n")
            
            profiler = None  # Clear profiler
    

if __name__ == '__main__':
    main()

    
