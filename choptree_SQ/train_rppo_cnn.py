import os
import gym
import minerl  # type: ignore
import torch
from stable_baselines3.common.env_util import make_vec_env  # type: ignore
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback  # type: ignore
from sb3_contrib import RecurrentPPO  # type: ignore

from custom_reward_wrapper import CustomRewardWrapperCNN
from wrappers import MultiDiscreteToDictActionWrapper
from bc_extractor_cnn import BCFeatureExtractorCNN

# Constants
INVENTORY_KEYS = ["log"]
MODEL_PATH = "models/rppo_bc_model_cnn"
BC_MODEL_PATH = "models/bc_model_cnn.pth"

# --- Policy Configuration ---
policy_kwargs = dict(
    features_extractor_class=BCFeatureExtractorCNN,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[64], vf=[64]),
    lstm_hidden_size=128,
    n_lstm_layers=1,
    shared_lstm=False
)

# --- Environment Setup ---
def make_env():
    env = gym.make("MineRLObtainDiamondShovel-v0")
    env = CustomRewardWrapperCNN(env)
    env = MultiDiscreteToDictActionWrapper(env)
    print(env.reset())
    print(env.observation_space)
    return env

vec_env = make_vec_env(make_env, n_envs=1)

# --- Callback for Early Stopping ---
class Stop(BaseCallback):
    def __init__(self, max_steps, verbose=0):
        super().__init__(verbose)
        self.max_steps = max_steps

    def _on_step(self) -> bool:
        return self.num_timesteps < self.max_steps

checkpoint_callback = CheckpointCallback(
    save_freq=4000,
    save_path="./checkpoints/",
    name_prefix="rppo_bc_cnn"
)

callback = CallbackList([
    Stop(max_steps=17500),
    checkpoint_callback
])

# --- Training Loop ---
def train_rppo_model():
    if os.path.exists(MODEL_PATH + ".zip"):
        print("\nLoading existing RPPO model...\n")
        model = RecurrentPPO.load(MODEL_PATH, env=vec_env, policy_kwargs=policy_kwargs, verbose=1)
    else:
        print("\nStarting new RPPO model...\n")
        model = RecurrentPPO(
            policy="MultiInputLstmPolicy",
            env=vec_env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            ent_coef=0.01
        )

    model.learn(
        total_timesteps=17500,
        callback=callback,
        reset_num_timesteps=True,
        progress_bar=True
    )

    model.save(MODEL_PATH)
    print(f"Recurrent PPO training done. Model saved to: {MODEL_PATH}")

# --- Entrypoint ---
def main():
    train_rppo_model()

if __name__ == "__main__":
    main()
