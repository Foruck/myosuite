""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

from myosuite.utils import gym; register=gym.register
from myosuite.envs.env_variants import register_env_variant

import os
import numpy as np

# utility to register envs with all muscle conditions
def register_env_with_variants(id, entry_point, max_episode_steps, kwargs):
    # register_env_with_variants base env
    register(
        id=id,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps,
        kwargs=kwargs
    )
    #register variants env with sarcopenia
    if id[:3] == "myo":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'sarcopenia'},
            variant_id=id[:3]+"Sarc"+id[3:],
            silent=True
        )
    #register variants with fatigue
    if id[:3] == "myo":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'fatigue'},
            variant_id=id[:3]+"Fati"+id[3:],
            silent=True
        )

    #register variants with tendon transfer
    if id[:7] == "myoHand":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'reafferentation'},
            variant_id=id[:3]+"Reaf"+id[3:],
            silent=True
        )

curr_dir = os.path.dirname(os.path.abspath(__file__))

print("MyoSuite:> Registering Myo Envs")

# Gait Torso Reaching ==============================
from myosuite.physics.sim_scene import SimBackend
sim_backend = SimBackend.get_sim_backend()

leg_model='/../../../simhive/myo_sim/leg/myolegs.xml'


# Gait Torso Walking ==============================
register_env_with_variants(id='myoLegIm-v0',
        entry_point='myosuite.envs.myo.myoim.legImitation_v0:legImitationEnvV0',
        max_episode_steps=1000,
        kwargs={
            'model_path': curr_dir + leg_model,
            'normalize_act': True,
            'min_height':0.8,    # minimum center of mass height before reset
            'max_rot':0.8,       # maximum rotation before reset
            'hip_period':100,    # desired periodic hip angle movement
            'reset_type':'init', # none, init, random
            'target_x_vel':0.0,  # desired x velocity in m/s
            'target_y_vel':1.2,  # desired y velocity in m/s
            'target_rot': None   # if None then the initial root pos will be taken, otherwise provide quat
            'ref_cand': '/ssd/xinpeng/myoref/refcand.pkl',
            'ref_path': '/ssd/xinpeng/myoref',
        }
    )
