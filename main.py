import gymnasium as gym
from bg_change import make
import numpy as np
import torch
import argparse
import os
from tqdm import tqdm

from utils.spatial_attention import visualize_spatial_attention
from utils.video_recorder import VideoRecorder
from utils.logger import Logger
from agent.ReplayBuffer import ReplayBuffer
from agent.DRIBOSacAgent import DRIBOSacAgent

def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image

def setup_cuda_optimization():
    """Configure CUDA/cuDNN for optimal DRIBO training performance"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        print("CUDA Optimizations Enabled")
    else:
        print("CUDA not available, running on CPU")

def parse_args():
    parser = argparse.ArgumentParser()

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Train the agent')
    mode_group.add_argument('--evaluate', action='store_true', help='Evaluate the agent')

    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--resume', default=None, action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--frame_skip', default=4, type=int)
    
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument('--visualize_attention', action='store_true', help='Visualize spatial attention maps')
    save_group.add_argument('--save_video',  action='store_true', help='Save video')


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



def setup_environment(args):
    """Common environment setup for both train and eval modes"""
    frame_skip = args.frame_skip
    frame_stack = 1
    img_size = 100
    augmented_img_size = 84
    total_frames = 1000
    
    base_dir = "/content/drive/MyDrive/DRIBO_logs"
    env_name = args.domain_name + '-' + args.task_name
    work_dir = os.path.join(base_dir, env_name)
    # work_dir =  "./log" + '/' + env_name
    
    os.makedirs(work_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_cuda_optimization()
    
    config = {
        'frame_skip': frame_skip,
        'frame_stack': frame_stack,
        'img_size': img_size,
        'augmented_img_size': augmented_img_size,
        'total_frames': total_frames,
        'work_dir': work_dir,
        'device': device,
        'env_name': env_name
    }
    
    return config



def train(args):
    """Training mode"""
    config = setup_environment(args)
    
    # Training-specific parameters
    replay_buffer_capacity = 50000
    resource_files = 'dataset/train/*.avi'
    eval_resource_files = 'dataset/test/*.avi'
    batch_size = 8
    episode_len = config['total_frames'] // config['frame_skip']
    num_train_steps = 500000
    eval_freq = 5000
    num_eval_episodes = 3
    init_step = 1000
    log_interval = 100

    # Create directories
    video_dir = os.path.join(config['work_dir'], 'video')
    model_dir = os.path.join(config['work_dir'], 'model')
    buffer_dir = os.path.join(config['work_dir'], 'buffer')
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(buffer_dir, exist_ok=True)

    # Create environments
    env = make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        resource_files=resource_files,
        total_frames=config['total_frames'],
        frame_skip=config['frame_skip'],
        frame_stack=config['frame_stack'],
        seed=42,
        width=config['img_size'],
        height=config['img_size'],
        render_mode=None,
        extra='train'
    )

    eval_env = make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        resource_files=eval_resource_files,
        total_frames=config['total_frames'],
        frame_skip=config['frame_skip'],
        frame_stack=config['frame_stack'],
        seed=42,
        width=640,
        height=480,
        render_mode="rgb_array",
        extra='eval',
    )

    video = VideoRecorder(dir_name=video_dir, height=480, width=640)

    # Shape definition
    action_shape = env.action_space.shape
    obs_shape = (3*config['frame_stack'], config['img_size'], config['img_size'])
    augmented_obs_shape = (3*config['frame_stack'], config['augmented_img_size'], config['augmented_img_size'])

    # Replay buffer
    replay_buffer = ReplayBuffer(
        obs_shape,
        action_shape,
        capacity=replay_buffer_capacity,
        batch_size=batch_size,
        episode_len=episode_len,
        device=config['device'],
        image_size=config['augmented_img_size']
    )

    # Agent
    agent = DRIBOSacAgent(
        obses_shape=augmented_obs_shape,
        actions_shape=action_shape,
        device=config['device']
    )
    
    # Load model
    start_step = 0
    if args.resume:
        if os.path.exists(model_dir) and os.listdir(model_dir):
            start_step = agent.load(model_dir)


            if os.path.exists(os.path.join(buffer_dir, 'replay_buffer.npz')):
                replay_buffer.load(buffer_dir)
                print(f"Loaded replay buffer")
            else:
                print(f"No replay buffer found, starting with empty buffer")
        else:
            print("No checkpoint found, starting from scratch")

    logger = Logger(config['work_dir'])
    episode, episode_reward, episode_step, terminated =0, 0, 0, True
    max_mean_ep_reward = 0

    pbar = tqdm(range(start_step, num_train_steps), desc="Training", initial=start_step, total=num_train_steps)

    for t in pbar:
        
        if t> init_step and t %  eval_freq == 0:
            logger.log('eval/episode', episode, t)

            all_ep_rewards = []
            for i in range(num_eval_episodes):
                obs_eval,_ = eval_env.reset()
                prev_state_eval = None
                prev_action_eval = None
                video.init(enabled=(i == 0))
                terminated_eval = False
                episode_reward_eval = 0
                while not terminated_eval:
                    # center crop image
                    obs_eval = center_crop_image(obs_eval, config['augmented_img_size'])
                    with eval_mode(agent):
                            action_eval, prev_action_eval, prev_state_eval = agent.select_action(obs_eval, prev_action_eval, prev_state_eval)
                    obs_eval, reward_eval, terminated_eval, truncated_eval, info_eval = eval_env.step(action_eval)
                    video.record(eval_env)
                    episode_reward_eval += reward_eval

                video.save('%d.mp4' % t)
                logger.log('eval/' + 'episode_reward', episode_reward_eval, t)
                all_ep_rewards.append(episode_reward_eval)

            mean_ep_reward = np.mean(all_ep_rewards)
            best_ep_reward = np.max(all_ep_rewards)
            logger.log('eval/' + 'mean_episode_reward', mean_ep_reward, t)
            logger.log('eval/' + 'best_episode_reward', best_ep_reward, t)
            logger.dump(t)

            if mean_ep_reward > max_mean_ep_reward:
                max_mean_ep_reward = mean_ep_reward
                agent.save(model_dir, t, episode)
                replay_buffer.save(buffer_dir)

        if terminated:
            if t > init_step and t % log_interval == 0:
                logger.dump(t)
            if t % log_interval == 0:
                logger.log('train/episode_reward', episode_reward, t)

            obs,_ = env.reset()
            prev_state = None
            prev_action = None
            terminated = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if t % log_interval == 0:
                logger.log('train/episode', episode, t)

        # Random exploration phase
        if t < init_step:
            action = env.action_space.sample()

        # Policy-based action selection
        else:
            with eval_mode(agent.encoder, agent.actor):
                action, prev_action, prev_state = agent.sample_action(obs, prev_action, prev_state)

        # Training updates
        if t >= init_step:
            agent.update(replay_buffer, logger, t)

        next_obs, reward, terminated, truncated, info = env.step(action)


        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(terminated) #done_bool = 0.0 if truncated else float(terminated)
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)
        
        obs = next_obs
        episode_step += 1



def evaluate(args):
    """Evaluation mode"""
    config = setup_environment(args)
    
    eval_resource_files = 'dataset/test/*.avi'
    num_eval_episodes = 5
    save_video = args.save_video
    visualize_attention = args.visualize_attention
    attention_freq = 125
    env_width, env_height = 100, 100

    if save_video:
        env_width, env_height = 640, 480
    elif visualize_attention:
        env_width, env_height = 100, 100

    video_dir = os.path.join(config['work_dir'], 'eval_video')
    model_dir = os.path.join(config['work_dir'], 'model')
    attention_dir = os.path.join(config['work_dir'], 'attention_maps')
    os.makedirs(video_dir, exist_ok=True)
    if visualize_attention:
        os.makedirs(attention_dir, exist_ok=True)

    eval_env = make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        resource_files=eval_resource_files,
        total_frames=config['total_frames'],
        frame_skip=config['frame_skip'],
        frame_stack=config['frame_stack'],
        seed=42,
        width=env_width,
        height=env_height,
        render_mode="rgb_array",
        extra='eval',
    )

    if save_video:
        video = VideoRecorder(dir_name=video_dir, height=480, width=640)
    # Shape definition
    action_shape = eval_env.action_space.shape
    augmented_obs_shape = (3*config['frame_stack'], config['augmented_img_size'], config['augmented_img_size'])

    # Agent
    agent = DRIBOSacAgent(
        obses_shape=augmented_obs_shape,
        actions_shape=action_shape,
        device=config['device']
    )

    #Load model
    if os.path.exists(model_dir) and os.listdir(model_dir):
        start_step = agent.load(model_dir)
    else:
        print(f"Error: No trained model found in {model_dir}")
        return

    # Run evaluation episodes
    all_ep_rewards = []
    all_ep_lengths = []
    
    print(f"\nRunning {num_eval_episodes} evaluation episodes...")
    
    for i in tqdm(range(num_eval_episodes), desc="Evaluating"):
        obs_eval, _ = eval_env.reset()
        prev_state_eval = None
        prev_action_eval = None
        
        if save_video:
            video.init(enabled=True)
        
        terminated_eval = False
        episode_reward_eval = 0
        episode_length = 0
        
        while not terminated_eval:
            obs_eval = center_crop_image(obs_eval, config['augmented_img_size'])
            
            if visualize_attention and episode_length == 0 :
                # Convert to torch tensor if needed
                obs_tensor = torch.from_numpy(obs_eval).float().to(config['device'])
                if obs_tensor.max() > 1.0:
                    obs_tensor = obs_tensor / 255.0
                
                visualize_spatial_attention(
                    agent.encoder, 
                    obs_tensor, 
                    episode=f"ep_{i+1}",
                    save_dir=attention_dir,
                )

            with eval_mode(agent):
                action_eval, prev_action_eval, prev_state_eval = agent.select_action(
                    obs_eval, prev_action_eval, prev_state_eval
                )
            
            obs_eval, reward_eval, terminated_eval, truncated_eval, info_eval = eval_env.step(action_eval)
            
            if save_video:
                video.record(eval_env)
            
            episode_reward_eval += reward_eval
            episode_length += 1

        if save_video:
            video.save(f'eval_episode_{i+1}.mp4')
        
        all_ep_rewards.append(episode_reward_eval)
        all_ep_lengths.append(episode_length)
        
        print(f"Episode {i+1}: Reward = {episode_reward_eval:.2f}, Length = {episode_length*config['frame_skip']}")

    # Print summary statistics
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Number of episodes: {num_eval_episodes}")
    print(f"Mean reward: {np.mean(all_ep_rewards):.2f} ± {np.std(all_ep_rewards):.2f}")
    print(f"Min reward: {np.min(all_ep_rewards):.2f}")
    print(f"Max reward: {np.max(all_ep_rewards):.2f}")
    print(f"Mean episode length: {np.mean(all_ep_lengths):.2f}")
    print("="*50)



def main():
    args = parse_args()
    
    if args.train:
        print("Starting training mode...")
        train(args)
    elif args.evaluate:
        print("Starting evaluation mode...")
        evaluate(args)



    # args = parse_args()
    # domain_name = args.domain_name
    # task_name = args.task_name
    # resume = args.resume
    # frame_skip=4
    # frame_stack=1
    # img_size = 100
    # augmented_img_size = 84
    # replay_buffer_capacity = 50000
    # resource_files = 'dataset/train/*.avi'
    # eval_resource_files = 'dataset/test/*.avi'
    # total_frames = 1000
    # save_video = True

    # batch_size = 8
    # episode_len = total_frames // frame_skip
    # num_train_steps = 400000
    # eval_freq = 5000
    # num_eval_episodes = 3
    # init_step = 1000
    # log_interval = 100


    # base_dir = "/content/drive/MyDrive/DRIBO_logs"

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # setup_cuda_optimization()

    # # Create environment
    # env = make(
    #     domain_name=domain_name,
    #     task_name=task_name,
    #     resource_files=resource_files,
    #     total_frames=total_frames,
    #     frame_skip=frame_skip,
    #     frame_stack=frame_stack,
    #     seed= 42,
    #     width= img_size,
    #     height= img_size,
    #     render_mode=None,
    #     extra='train'
    # )

    # eval_env = make(
    #     domain_name=domain_name,
    #     task_name=task_name,
    #     resource_files=eval_resource_files,
    #     total_frames=total_frames,
    #     frame_skip=frame_skip,
    #     frame_stack=frame_stack,
    #     seed= 42,
    #     width= 640,
    #     height= 480,
    #     render_mode="rgb_array",
    #     extra='eval',
    # )

    # # make directory
    # env_name = domain_name + '-' + task_name
    
    # # work_dir =  "./log" + '/' + env_name
    # work_dir = os.path.join(base_dir, env_name)
    
    # os.makedirs(work_dir, exist_ok=True)
    # video_dir = os.path.join(work_dir, 'video')
    # os.makedirs(video_dir, exist_ok=True)
    # model_dir = os.path.join(work_dir, 'model')
    # os.makedirs(model_dir, exist_ok=True)
    # buffer_dir = os.path.join(work_dir, 'buffer')
    # os.makedirs(buffer_dir, exist_ok=True)

    # if(save_video):
    #     video = VideoRecorder(dir_name=video_dir, height=480, width=640)

    # # Shape definition
    # action_shape = env.action_space.shape
    # obs_shape = (3*frame_stack, img_size, img_size)
    # augmented_obs_shape = (3*frame_stack, augmented_img_size, augmented_img_size)
    # print(action_shape)

    # # Replay buffer
    # replay_buffer = ReplayBuffer(
    #     obs_shape,
    #     action_shape,
    #     capacity = replay_buffer_capacity,
    #     batch_size = batch_size,
    #     episode_len = episode_len,
    #     device= device,
    #     image_size=augmented_img_size
    # )

    # # Agent
    # agent = DRIBOSacAgent(
    #     obses_shape= augmented_obs_shape,
    #     actions_shape= action_shape,
    #     device= device
    # )


    # # Load model
    # start_step = 0
    # if resume is not None:
    #     if os.path.exists(model_dir) and os.listdir(model_dir):
    #         start_step = agent.load(model_dir)


    #         if os.path.exists(os.path.join(buffer_dir, 'replay_buffer.npz')):
    #             replay_buffer.load(buffer_dir)
    #             print(f"Loaded replay buffer")
    #         else:
    #             print(f"No replay buffer found, starting with empty buffer")
    #     else:
    #         print("No checkpoint found, starting from scratch")

    # logger = Logger(work_dir)
    # episode, episode_reward, episode_step, terminated =0, 0, 0, True
    # max_mean_ep_reward = 0

    # pbar = tqdm(range(start_step, num_train_steps), desc="Training", initial=start_step, total=num_train_steps)

    # for t in pbar:
        
    #     if t> init_step and t %  eval_freq == 0:
    #         logger.log('eval/episode', episode, t)

    #         all_ep_rewards = []
    #         for i in range(num_eval_episodes):
    #             obs_eval,_ = eval_env.reset()
    #             prev_state_eval = None
    #             prev_action_eval = None
    #             video.init(enabled=(i == 0))
    #             terminated_eval = False
    #             episode_reward_eval = 0
    #             while not terminated_eval:
    #                 # center crop image
    #                 obs_eval = center_crop_image(obs_eval, augmented_img_size)
    #                 with eval_mode(agent):
    #                         action_eval, prev_action_eval, prev_state_eval = agent.select_action(obs_eval, prev_action_eval, prev_state_eval)
    #                 obs_eval, reward_eval, terminated_eval, truncated_eval, info_eval = eval_env.step(action_eval)
    #                 video.record(eval_env)
    #                 episode_reward_eval += reward_eval

    #             video.save('%d.mp4' % t)
    #             logger.log('eval/' + 'episode_reward', episode_reward_eval, t)
    #             all_ep_rewards.append(episode_reward_eval)

    #         mean_ep_reward = np.mean(all_ep_rewards)
    #         best_ep_reward = np.max(all_ep_rewards)
    #         logger.log('eval/' + 'mean_episode_reward', mean_ep_reward, t)
    #         logger.log('eval/' + 'best_episode_reward', best_ep_reward, t)
    #         logger.dump(t)

    #         if mean_ep_reward > max_mean_ep_reward:
    #             max_mean_ep_reward = mean_ep_reward
    #             agent.save(model_dir, t, episode)
    #             replay_buffer.save(buffer_dir)

    #     if terminated:
    #         if t > init_step and t % log_interval == 0:
    #             logger.dump(t)
    #         if t % log_interval == 0:
    #             logger.log('train/episode_reward', episode_reward, t)

    #         obs,_ = env.reset()
    #         prev_state = None
    #         prev_action = None
    #         terminated = False
    #         episode_reward = 0
    #         episode_step = 0
    #         episode += 1
    #         if t % log_interval == 0:
    #             logger.log('train/episode', episode, t)

    #     # Random exploration phase
    #     if t < init_step:
    #         action = env.action_space.sample()

    #     # Policy-based action selection
    #     else:
    #         with eval_mode(agent.encoder, agent.actor):
    #             action, prev_action, prev_state = agent.sample_action(obs, prev_action, prev_state)

    #     # Training updates
    #     if t >= init_step:
    #         agent.update(replay_buffer, logger, t)

    #     next_obs, reward, terminated, truncated, info = env.step(action)


    #     done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(terminated)
    #     episode_reward += reward
    #     replay_buffer.add(obs, action, reward, next_obs, done_bool)
        
    #     obs = next_obs
    #     episode_step += 1

        
if __name__ == '__main__':
    main()