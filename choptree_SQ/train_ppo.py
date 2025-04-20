import gym
import torch
import numpy as np
import cv2
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from custom_reward_wrapper import CustomRewardWrapper
from wrappers import FlattenObservationWrapper, MultiDiscreteToDictActionWrapper
from bc_extractor import BCFeatureExtractor
import os
import minerl

# Constants
INVENTORY_KEYS = ["log"]
CAMERA_BINS = 21
CAMERA_CENTER = CAMERA_BINS // 2
INPUT_DIM = 64 * 64 * 3 + len(INVENTORY_KEYS)
OUTPUT_DIM = 6 + 2  # 6 binary buttons + 2 camera bins
MODEL_PATH = "models/ppo_bc_model"
BC_MODEL_PATH = "models/bc_model.pth"

# # Load BC model as feature extractor
# class BCFeatureExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space, features_dim=128):
#         super().__init__(observation_space, features_dim)
#         self.net = PolicyNetwork(INPUT_DIM, OUTPUT_DIM)
#         self.net.load_state_dict(torch.load(BC_MODEL_PATH))
#         self.net.eval()

#     def forward(self, x):
#         return self.net.fc2(torch.relu(self.net.fc1(x)))

policy_kwargs = dict(
    features_extractor_class=BCFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[64], vf=[64])
)

# Make environment
def make_env():
    env = gym.make("MineRLObtainDiamondShovel-v0")
    env = CustomRewardWrapper(env)
    env = FlattenObservationWrapper(env)
    env = MultiDiscreteToDictActionWrapper(env)
    return env

vec_env = make_vec_env(make_env, n_envs=1)

# stop training
class Stop(BaseCallback):
    def __init__(self, max_steps, verbose=0):
        super().__init__(verbose)
        self.max_steps = max_steps

    def _on_step(self) -> bool:
        return self.num_timesteps < self.max_steps

# save regularly 
checkpoint_callback = CheckpointCallback(
    save_freq=4000,
    save_path="./checkpoints/",
    name_prefix="ppo_bc"
)

callback = CallbackList([
    Stop(max_steps=17500),
    checkpoint_callback
])

# Train PPO starting from BC model
def train_PPO_model():
    if os.path.exists(MODEL_PATH + ".zip"):
        print("\nLoading existing PPO model...\n")
        model = PPO.load(MODEL_PATH, env=vec_env, policy_kwargs=policy_kwargs, verbose=1)
    else:
        print("\nStarting new PPO model...\n")
        model = PPO("MlpPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs)

    model.learn(
        total_timesteps=17500,
        callback=callback,
        reset_num_timesteps=True,
        progress_bar=True
    )

    model.save(MODEL_PATH)
    print(f"PPO fine-tuning done. Model saved to: {MODEL_PATH}")

def main():
    train_PPO_model()

if __name__ == "__main__":
    main()

