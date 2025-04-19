
import gym
import torch
import numpy as np
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from custom_reward_wrapper import CustomRewardWrapper
from wrappers import FlattenObservationWrapper, MultiDiscreteToDictActionWrapper
from train_bc import PolicyNetwork
from bc_extractor import BCFeatureExtractor
import minerl

# --- Constants ---
INVENTORY_KEYS = ["log"]
CAMERA_BINS = 21
CAMERA_CENTER = CAMERA_BINS // 2
INPUT_DIM = 64 * 64 * 3 + len(INVENTORY_KEYS)
OUTPUT_DIM = 6 + 2  # 6 discrete buttons + 2 camera bins

def make_env():
    env = gym.make("MineRLObtainDiamondShovel-v0")
    env = CustomRewardWrapper(env)
    env = FlattenObservationWrapper(env)
    env = MultiDiscreteToDictActionWrapper(env)
    return env

vec_env = make_vec_env(make_env, n_envs=1)

# --- Callbacks ---
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path="./checkpoints/",
    name_prefix="softq_bc"
)

# --- Policy config ---
policy_kwargs = dict(
    features_extractor_class=BCFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=[256, 256]
)

# --- Train Soft Q (SAC for discrete MultiDiscrete actions) ---
model = SAC(
    policy="MlpPolicy",
    env=vec_env,
    verbose=1,
    policy_kwargs=policy_kwargs,
    tensorboard_log="./tb_logs"
)

model.learn(total_timesteps=17500, callback=checkpoint_callback)
model.save("models/softq_bc")
print("Soft Q fine-tuning done. Model saved to: softq_finetuned_bc")

