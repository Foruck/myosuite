import mujoco
import myosuite
from myosuite.utils import gym
import skvideo.io
import numpy as np
import os
import tqdm
import warnings
warnings.filterwarnings(action='once')
import deprl

# myoLegWalk-v0
# myoChallengeRunTrackP1-v0
env_string = 'myoLegWalk-v0' 
ckpt_path  = '/Disk4/xinpeng/myosuite/myosuite/agents/baselines_DEPRL/myoLeg/240813.162023/'
video_name = 'Run_deprl'
env = gym.make(env_string, reset_type='init')
policy = deprl.load(ckpt_path, env)

frames = []
N = 5 # number of episodes
for i in tqdm.trange(N):
    ep_rewards = []
    done = False
    obs = env.reset()[0]
    for j in tqdm.trange(50 * 12):
        geom_1_indices = np.where(env.sim.model.geom_group == 1)
        env.sim.model.geom_rgba[geom_1_indices, 3] = 0
        frame = env.sim.renderer.render_offscreen(
                          width=400,
                          height=400,
                          camera_id=1)
        frames.append(frame)
        action = policy(obs)
        obs, reward, done, _, info = env.step(action)
        if done:
            break

os.makedirs('videos', exist_ok=True)
# make a local copy
skvideo.io.vwrite(f'videos/{video_name}.mp4', np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})