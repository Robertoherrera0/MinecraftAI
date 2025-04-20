import gym
import torch
import numpy as np
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from custom_reward_wrapper import CustomRewardWrapper
from wrappers import FlattenObservationWrapper, MultiDiscreteToDictActionWrapper
from train_bc import PolicyNetwork
from bc_extractor import BCFeatureExtractor
import os
import minerl

# Constants
INVENTORY_KEYS = ["log"]
CAMERA_BINS = 21
CAMERA_CENTER = CAMERA_BINS // 2
INPUT_DIM = 64 * 64 * 3 + len(INVENTORY_KEYS)
OUTPUT_DIM = 6 + 2  # 6 discrete buttons + 2 camera bins
MODEL_PATH = "models/softq_bc_model"
BC_MODEL_PATH = "models/bc_model.pth"

policy_kwargs = dict(
    features_extractor_class=BCFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=[256, 256]
)

# Make environment
def make_env():
    env = gym.make("MineRLObtainDiamondShovel-v0")
    env = CustomRewardWrapper(env)
    env = FlattenObservationWrapper(env)
    env = MultiDiscreteToDictActionWrapper(env)
    return env

vec_env = make_vec_env(make_env, n_envs=1)

# Stop training
class Stop(BaseCallback):
    def __init__(self, max_steps, verbose=0):
        super().__init__(verbose)
        self.max_steps = max_steps

    def _on_step(self) -> bool:
        return self.num_timesteps < self.max_steps

# Save regularly
checkpoint_callback = CheckpointCallback(
    save_freq=4000,
    save_path="./checkpoints/",
    name_prefix="softq_bc"
)

callback = CallbackList([
    Stop(max_steps=17500),
    checkpoint_callback
])

# Train Soft Q from BC
def train_softq_model():
    if os.path.exists(MODEL_PATH + ".zip"):
        print("\nLoading existing Soft Q model...\n")
        model = SAC.load(MODEL_PATH, env=vec_env, policy_kwargs=policy_kwargs, verbose=1)
    else:
        print("\nStarting new Soft Q model...\n")
        model = SAC("MlpPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./tb_logs")

    model.learn(
        total_timesteps=17500,
        callback=callback,
        reset_num_timesteps=True,
        progress_bar=True
    )

    model.save(MODEL_PATH)
    print(f"Soft Q fine-tuning done. Model saved to: {MODEL_PATH}")

def main():
    train_softq_model()

if __name__ == "__main__":
    main()
