import gym
from stable_baselines3 import PPO  # type: ignore
from stable_baselines3.common.env_util import make_vec_env  # type: ignore
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback  # type: ignore

from wrappers.custom_reward_wrapper import CustomRewardWrapper
from wrappers.wrappers import MultiDiscreteToDictActionWrapper
from features.bc_extractor import BCFeatureExtractor
import os
import minerl  # type: ignore

# Paths
MODEL_PATH = "../models/ppo_bc_model"
BC_MODEL_PATH = "../models/bc_model.pth"

# PPO Policy Settings
policy_kwargs = dict(
    features_extractor_class=BCFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[64], vf=[64])
)

# Make Environment
def make_env():
    env = gym.make("MineRLObtainDiamondShovel-v0")
    env = CustomRewardWrapper(env)
    env = MultiDiscreteToDictActionWrapper(env)
    return env

vec_env = make_vec_env(make_env, n_envs=1)

# Stop Training Callback
class Stop(BaseCallback):
    def __init__(self, max_steps, verbose=0):
        super().__init__(verbose)
        self.max_steps = max_steps

    def _on_step(self) -> bool:
        return self.num_timesteps < self.max_steps

# Save Checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=4000,
    save_path="../models/checkpoints/",
    name_prefix="ppo_bc"
)

callback = CallbackList([
    Stop(max_steps=17500),
    checkpoint_callback
])

# Train PPO with BC Feature Extractor
def train_ppo_model():
    if os.path.exists(MODEL_PATH + ".zip"):
        print("\nLoading existing PPO model...\n")
        model = PPO.load(MODEL_PATH, env=vec_env, policy_kwargs=policy_kwargs, verbose=1)
    else:
        print("\nStarting new PPO model...\n")
        model = PPO(
            policy="MultiInputPolicy",  # <- supports dict observations
            env=vec_env,
            verbose=1,
            policy_kwargs=policy_kwargs
        )

    model.learn(
        total_timesteps=17500,
        callback=callback,
        reset_num_timesteps=True,
        progress_bar=True
    )

    model.save(MODEL_PATH)
    print(f"PPO fine-tuning done. Model saved to: {MODEL_PATH}")

def main():
    train_ppo_model()

if __name__ == "__main__":
    main()
