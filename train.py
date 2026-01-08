import gymnasium as gym
from bg_change import make
import numpy as np
import cv2

# Create environment WITH rendering enabled
env = make(
    domain_name='cheetah',
    task_name='run',
    resource_files='dataset/train/*.avi',
    img_source='video',
    visualize_reward=False,
    total_frames=1000,
    seed=42,
    from_pixels=True,
    render_mode="rgb_array",   # 🔑 REQUIRED
)

obs, info = env.reset(seed=42)

for _ in range(300):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    frame = env.render()  # (H, W, 3), uint8
    cv2.imshow("DMC + Video Background", frame[:, :, ::-1])  # RGB → BGR

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if terminated or truncated:
        obs, info = env.reset()

env.close()
cv2.destroyAllWindows()
