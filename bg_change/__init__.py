import gymnasium as gym


def make(
    domain_name,
    task_name,
    resource_files,
    img_source,
    total_frames=1000,
    seed=1,
    visualize_reward=True,
    from_pixels=False,
    height=480,
    width=640,
    camera_id=0,
    frame_skip=1,
    episode_length=1000,
    frame_stack=1,
    environment_kwargs=None,
    extra='train',
    render_mode=None,
):
    """
    Create a Gymnasium-compatible DMC environment with video backgrounds.
    
    Changes from Gym version:
    - Uses gymnasium.register instead of gym.envs.registration.register
    - Added render_mode parameter for Gymnasium render API
    - Check registry with 'in gym.envs.registry' instead of 'in gym.envs.registry.env_specs'
    """
    env_id = f'dmc_{domain_name}_{task_name}_{extra}_{seed}-v1'

    if from_pixels:
        assert not visualize_reward, \
            'cannot use visualize reward when learning from pixels'

    # shorten episode length
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    # Check if environment is already registered
    if env_id not in gym.envs.registry:
        gym.register(
            id=env_id,
            entry_point='bg_change.wrappers:DMCWrapper',
            max_episode_steps=max_episode_steps,
            kwargs={
                'domain_name': domain_name,
                'task_name': task_name,
                'resource_files': resource_files,
                'img_source': img_source,
                'total_frames': total_frames,
                'task_kwargs': {
                    'random': seed
                },
                'environment_kwargs': environment_kwargs,
                'visualize_reward': visualize_reward,
                'from_pixels': from_pixels,
                'height': height,
                'width': width,
                'camera_id': camera_id,
                'frame_skip': frame_skip,
                'frame_stack': frame_stack,
                'render_mode': render_mode,
            },
        )
    
    return gym.make(env_id, render_mode=render_mode)


# Example usage
if __name__ == "__main__":
    # Create environment with video backgrounds
    env = make(
        domain_name='cartpole',
        task_name='swingup',
        resource_files='/path/to/kinetics/*.mp4',
        img_source='video',
        total_frames=1000,
        seed=42,
        from_pixels=True,
        height=84,
        width=84,
        frame_skip=4,
        extra='train',
        render_mode='rgb_array'
    )
    
    # Use with Gymnasium API
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Info: {info}")
    
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()