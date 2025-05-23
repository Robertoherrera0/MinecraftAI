import gym
from stable_baselines3 import PPO  # type:ignore
from stable_baselines3.common.env_util import make_vec_env  # type:ignore
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback  # type:ignore

from wrappers.custom_reward_wrapper import CustomRewardWrapper
from wrappers.wrappers import FlattenObservationWrapper, MultiDiscreteToDictActionWrapper
from features.bc_extractor import BCFeatureExtractor
import os
import minerl  # type:ignore

# Constants
INVENTORY_KEYS = ["log"]
CAMERA_BINS = 21
CAMERA_CENTER = CAMERA_BINS // 2
INPUT_DIM = 64 * 64 * 3 + len(INVENTORY_KEYS)
OUTPUT_DIM = 6 + 2  # 6 binary buttons + 2 camera bins
MODEL_PATH = "../models/ppo_bc_model"
BC_MODEL_PATH = "../models/bc_model.pth"

# Use pre-trained BC feature extractor with small policy network
policy_kwargs = dict(
    features_extractor_class=BCFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[64], vf=[64])
)

# Create and wrap the MineRL environment
def make_env():
    env = gym.make("MineRLObtainDiamondShovel-v0")
    env = CustomRewardWrapper(env)
    env = FlattenObservationWrapper(env)
    env = MultiDiscreteToDictActionWrapper(env)
    return env

# Vectorized environment
vec_env = make_vec_env(make_env, n_envs=1)

# Stop training after fixed number of timesteps
class Stop(BaseCallback):
    def __init__(self, max_steps, verbose=0):
        super().__init__(verbose)
        self.max_steps = max_steps

    def _on_step(self) -> bool:
        return self.num_timesteps < self.max_steps

# Save model checkpoints during training
checkpoint_callback = CheckpointCallback(
    save_freq=4000,
    save_path="../models/checkpoints/",
    name_prefix="ppo_bc"
)

# Combine stop and checkpoint callbacks
callback = CallbackList([
    Stop(max_steps=17500),
    checkpoint_callback
])

# Train or load PPO model (using BC features)
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

# Entry point
def main():
    train_PPO_model()

if __name__ == "__main__":
    main()
