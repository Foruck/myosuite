import mujoco
import myosuite
from myosuite.utils import gym
import skvideo.io
import numpy as np
import os
from stable_baselines3 import PPO
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='once')
# myoLegWalk-v0
# myoChallengeRunTrackP1-v0
env_string = 'myoChallengeRunTrackP1-v0' 
ckpt_path  = '/Disk4/xinpeng/myosuite/myosuite/agents/outputs/2024-08-08/17-30-43/restore_checkpoint.zip'
video_name = 'chalRun'
env = gym.make(env_string, reset_type='init')

model = PPO("MlpPolicy", env, verbose=0).load(ckpt_path)

# Render trained policy
frames = []
all_rewards = []
for _ in tqdm(range(1)): # Randomization over different terrain types
  env.reset()
  ep_rewards = []
  done = False
  obs = env.reset()
  while not done:
      obs = env.obsdict2obsvec(env.obs_dict, env.obs_keys)[1]
      # get the next action from the policy
      action, _ = model.predict(obs, deterministic=True)
      geom_1_indices = np.where(env.sim.model.geom_group == 1)
      env.sim.model.geom_rgba[geom_1_indices, 3] = 0
      frame = env.sim.renderer.render_offscreen(
                        width=400,
                        height=400,
                        camera_id=1)
      frames.append(frame)
      # take an action based on the current observation
      obs, reward, done, info, _ = env.step(action)
      ep_rewards.append(reward)
  all_rewards.append(np.sum(ep_rewards))
print(f"Average reward: {np.mean(all_rewards)} over 5 episodes")
env.close()

os.makedirs('videos', exist_ok=True)
# make a local copy
skvideo.io.vwrite(f'videos/{video_name}.mp4', np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
