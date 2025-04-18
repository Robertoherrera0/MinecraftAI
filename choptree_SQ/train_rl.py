import gym
import torch
import numpy as np
import cv2
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from train_bc import PolicyNetwork
from custom_reward_wrapper import CustomRewardWrapper
from wrappers import FlattenObservationWrapper, MultiDiscreteToDictActionWrapper
import minerl 
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,  # save every 10k steps
    save_path="./checkpoints/",
    name_prefix="ppo_bc"
)

# --- Constants ---
INVENTORY_KEYS = ["log"]
CAMERA_BINS = 21
CAMERA_CENTER = CAMERA_BINS // 2
INPUT_DIM = 64 * 64 * 3 + len(INVENTORY_KEYS)
OUTPUT_DIM = 6 + 2  # 6 binary buttons + 2 camera bins

# --- Feature extractor using your BC model weights ---
class BCFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.net = PolicyNetwork(INPUT_DIM, OUTPUT_DIM)
        self.net.load_state_dict(torch.load("bc_model.pth"))
        self.net.eval()

    def forward(self, x):
        return self.net.fc2(torch.relu(self.net.fc1(x)))

# --- Build SB3 policy from BC model ---
policy_kwargs = dict(
    features_extractor_class=BCFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=[dict(pi=[64], vf=[64])]
)

def make_env():
    env = gym.make("MineRLObtainDiamondShovel-v0")
    env = CustomRewardWrapper(env)
    env = FlattenObservationWrapper(env)
    env = MultiDiscreteToDictActionWrapper(env)
    return env

vec_env = make_vec_env(make_env, n_envs=1)

# --- Train PPO starting from BC model ---
model = PPO("MlpPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs)

model.learn(total_timesteps=25000, callback=checkpoint_callback)

model.save("ppo_finetuned_bc")

print("PPO fine-tuning done. Model saved to: ppo_finetuned_bc")
