import os
import argparse
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from bg_change import make
import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.video_recorder import VideoRecorder
from agent.dreamer import Dreamer

def load_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image

def setup_cuda_optimization():
    """Configure CUDA/cuDNN for optimal training performance"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        print("CUDA Optimizations Enabled")
    else:
        print("CUDA not available, running on CPU")

def parse_args():
    parser = argparse.ArgumentParser(description='Dreamer Agent Training and Evaluation')

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Train the agent')
    mode_group.add_argument('--evaluate', action='store_true', help='Evaluate the agent')

    parser.add_argument('--config', type=str, default='./config/cheetah-run.yml',
                       help='Path to config file')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume training from latest checkpoint')
    parser.add_argument('--load_best', action='store_true',
                       help='Load best model (for evaluation)')
    parser.add_argument('--save_video', action='store_true', 
                       help='Save evaluation videos')
    parser.add_argument('--num_eval_episodes', type=int, default=5,
                       help='Number of episodes for evaluation')

    args = parser.parse_args()
    return args

def create_environment(config, mode='train'):
    """Create training or evaluation environment"""
    env_config = config['environment']
    
    if mode == 'train':
        env = make(
            domain_name=env_config['domain_name'],
            task_name=env_config['task_name'],
            resource_files=env_config['resource_files'],
            total_frames=env_config['total_frames'],
            frame_skip=env_config['frame_skip'],
            frame_stack=1,
            seed=env_config['seed'],
            width=env_config['width'],
            height=env_config['height'],
            render_mode=None,
            extra='train'
        )
    else:  # eval mode
        eval_resource_files = env_config.get('eval_resource_files', 
                                            env_config['resource_files'].replace('train', 'test'))
        env = make(
            domain_name=env_config['domain_name'],
            task_name=env_config['task_name'],
            resource_files=eval_resource_files,
            total_frames=env_config['total_frames'],
            frame_skip=env_config['frame_skip'],
            frame_stack=1,
            seed=env_config['seed'],
            width=640,
            height=480,
            render_mode="rgb_array",
            extra='eval'
        )
    
    return env

def train(config, args):
    """Training mode"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_cuda_optimization()
    
    # Create directories
    base_dir = "/content/drive/MyDrive/dreamer_logs"
    env_name = config['environment']['domain_name'] + '-' + config['environment']['task_name']
    work_dir = os.path.join(base_dir, env_name)
    
    os.makedirs(work_dir, exist_ok=True)
    video_dir = os.path.join(work_dir, 'video')
    model_dir = os.path.join(work_dir, 'model')
    buffer_dir = os.path.join(work_dir, 'buffer')
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(buffer_dir, exist_ok=True)

    writer = SummaryWriter(work_dir)

    # Create environments
    print("Creating training environment...")
    env = create_environment(config, mode='train')
    
    print("Creating evaluation environment...")
    eval_env = create_environment(config, mode='eval')

    obs_shape = (3, 84, 84)
    discrete_action_bool = False
    action_size = env.action_space.shape[0]
    
    print(f"Observation shape: {obs_shape}")
    print(f"Action size: {action_size}")
    print(f"Action space: Continuous")

    # Create agent
    agent = Dreamer(
        obs_shape, 
        discrete_action_bool, 
        action_size, 
        writer, 
        device, 
        config, 
        work_dir
    )

    # Resume training if requested
    start_step = 0
    if args.resume:
        if os.path.exists(model_dir) and os.listdir(model_dir):
            start_step = agent.load(work_dir, load_best=False)
            print(f"Resumed training from step {start_step}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Train
    print("\nStarting training...")
    agent.train(env, eval_env)
    
    # Save final model
    print("\nSaving final model...")
    agent.save(work_dir, agent.global_step, episode=agent.num_total_episode)
    
    print("Training complete!")

def evaluate(config, args):
    """Evaluation mode"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup directories
    env_name = config['environment']['domain_name'] + '-' + config['environment']['task_name']
    work_dir = config.get('work_dir', os.path.join("./log", env_name))
    
    model_dir = os.path.join(work_dir, 'model')
    eval_video_dir = os.path.join(work_dir, 'eval_videos')
    os.makedirs(eval_video_dir, exist_ok=True)

    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = create_environment(config, mode='eval')

    obs_shape = (3, 84, 84)
    discrete_action_bool = False
    action_size = eval_env.action_space.shape[0]

    # Create TensorBoard writer for evaluation logs
    writer = SummaryWriter(os.path.join(work_dir, 'eval_logs'))

    # Create agent
    agent = Dreamer(
        obs_shape, 
        discrete_action_bool, 
        action_size, 
        writer, 
        device, 
        config, 
        work_dir
    )

    # Load model
    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        print(f"Error: No trained model found in {model_dir}")
        return
    
    agent.load(work_dir, load_best=args.load_best)
    
    if args.load_best:
        print("Loaded best model for evaluation")
    else:
        print("Loaded latest model for evaluation")

    # Setup video recording
    if args.save_video:
        video = VideoRecorder(dir_name=eval_video_dir, height=480, width=640)
        agent.video = video
        print(f"Video recording enabled. Videos will be saved to {eval_video_dir}")

    # Run evaluation
    print(f"\nRunning {args.num_eval_episodes} evaluation episodes...")
    
    all_ep_rewards = []
    all_ep_lengths = []
    
    for ep in tqdm(range(args.num_eval_episodes), desc="Evaluating"):
        prev_state = None
        prev_action = None
        obs_raw, _ = eval_env.reset()
        
        if args.save_video:
            video.init(enabled=True)
        
        episode_reward = 0
        episode_length = 0
        terminated = False
        
        while not terminated:
            # Preprocess observation
            if obs_raw.shape[-1] != 84:
                obs = center_crop_image(obs_raw, 84)
            else:
                obs = obs_raw
            
            obs = torch.FloatTensor(obs).to(device).unsqueeze(0)
            
            # Get action (deterministic for evaluation)
            with torch.no_grad():
                state = agent.rssm(
                    obs=obs,
                    prev_action=prev_action,
                    prev_state=prev_state
                )
                
                action = agent.actor(state.stoch, state.det, deterministic_action=True)
                env_action = action.cpu().numpy()[0]
            
            # Step environment
            next_obs, reward, terminated, truncated, info = eval_env.step(env_action)
            
            if args.save_video:
                video.record(eval_env)
            
            episode_reward += reward
            episode_length += 1
            
            obs_raw = next_obs
            prev_state = state
            prev_action = action
            
            if truncated:
                terminated = True
        
        # Save video
        if args.save_video:
            video.save(f'eval_episode_{ep+1}.mp4')
        
        all_ep_rewards.append(episode_reward)
        all_ep_lengths.append(episode_length)
        
        print(f"Episode {ep+1}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    # Print summary statistics
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Number of episodes: {args.num_eval_episodes}")
    print(f"Mean reward: {np.mean(all_ep_rewards):.2f} ± {np.std(all_ep_rewards):.2f}")
    print(f"Min reward: {np.min(all_ep_rewards):.2f}")
    print(f"Max reward: {np.max(all_ep_rewards):.2f}")
    print(f"Mean episode length: {np.mean(all_ep_lengths):.2f}")
    print("="*60)
    
    # Log to TensorBoard
    writer.add_scalar("final_eval/mean_reward", np.mean(all_ep_rewards), 0)
    writer.add_scalar("final_eval/std_reward", np.std(all_ep_rewards), 0)
    writer.add_scalar("final_eval/max_reward", np.max(all_ep_rewards), 0)
    writer.add_scalar("final_eval/min_reward", np.min(all_ep_rewards), 0)

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Override config with command line arguments if needed
    if hasattr(args, 'num_eval_episodes'):
        if 'parameters' not in config:
            config['parameters'] = {}
        if 'dreamer' not in config['parameters']:
            config['parameters']['dreamer'] = {}
        config['parameters']['dreamer']['num_eval_episodes'] = args.num_eval_episodes
    
    if args.train:
        print("="*60)
        print("DREAMER TRAINING MODE")
        print("="*60)
        train(config, args)
    elif args.evaluate:
        print("="*60)
        print("DREAMER EVALUATION MODE")
        print("="*60)
        evaluate(config, args)

if __name__ == "__main__":
    main()