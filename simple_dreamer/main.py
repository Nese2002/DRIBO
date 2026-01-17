import os
import argparse
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from bg_change import make
import numpy as np
import torch
import argparse
import os
from tqdm import tqdm
import yaml

from utils.video_recorder import VideoRecorder
from utils.logger import Logger
# from agent.ReplayBuffer import ReplayBuffer

from agent.dreamer import Dreamer
# from dreamer.utils.utils import load_config, get_base_directory

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

    # parser.add_argument('--domain_name', default='cheetah')
    # parser.add_argument('--task_name', default='run')
    # parser.add_argument('--resume', default=None, action='store_true', help='Resume from latest checkpoint')
    # parser.add_argument('--frame_skip', default=4, type=int)
    
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument('--visualize_attention', action='store_true', help='Visualize spatial attention maps')
    save_group.add_argument('--save_video',  action='store_true', help='Save video')


    args = parser.parse_args()
    return args

def main(config_file):
    config = load_config(config_file)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create directories
    base_dir = "/content/drive/MyDrive/DRIBO_logs"
    env_name = config['environment']['domain_name'] + '-' + config['environment']['task_name']
    # work_dir = os.path.join(base_dir, env_name)
    work_dir =  "./log" + '/' + env_name
    
    os.makedirs(work_dir, exist_ok=True)
    video_dir = os.path.join(work_dir, 'video')
    model_dir = os.path.join(work_dir, 'model')
    buffer_dir = os.path.join(work_dir, 'buffer')
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(buffer_dir, exist_ok=True)

    writer = SummaryWriter(work_dir)


    env_config = config['environment']
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

    obs_shape = (3,84,84)#env.observation_space.shape
    print(obs_shape)
    # if isinstance(env.action_space, gym.spaces.Discrete):
    #     discrete_action_bool = True
    #     action_size = env.action_space.n
    #     print("discrete?")
    # elif isinstance(env.action_space, gym.spaces.Box):
    discrete_action_bool = False
    action_size = env.action_space.shape[0]
    print("Continueee")
    # else:
    #     raise Exception


    agent = Dreamer(
            obs_shape, discrete_action_bool, action_size, writer, device, config
        )

    agent.train(env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./config/cheetah-run.yml",
        help="config file to run(default: cheetah-run.yml)",
    )
    main(parser.parse_args().config)