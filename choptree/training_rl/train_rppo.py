import gym
import os
from stable_baselines3.common.env_util import make_vec_env  # type:ignore
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback  # type:ignore
from sb3_contrib import RecurrentPPO  # type:ignore

from choptree.wrappers.custom_reward_wrapper import CustomRewardWrapperRPPO
from choptree.wrappers.wrappers import FlattenObservationWrapper, MultiDiscreteToDictActionWrapper
from choptree.features.bc_extractor import BCFeatureExtractor
import minerl  # type:ignore

# Constants
INVENTORY_KEYS = ["log"]
CAMERA_BINS = 21
CAMERA_CENTER = CAMERA_BINS // 2
INPUT_DIM = 64 * 64 * 3 + len(INVENTORY_KEYS)
OUTPUT_DIM = 6 + 2
MODEL_PATH = "../models/rppo_bc_model"
BC_MODEL_PATH = "../models/bc_model.pth"

# RPPO policy configuration using BC feature extractor
policy_kwargs = dict(
    features_extractor_class=BCFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[64], vf=[64]),
    lstm_hidden_size=128,
    n_lstm_layers=1,
    shared_lstm=False
)

# Wrap the MineRL environment for RPPO
def make_env():
    env = gym.make("MineRLObtainDiamondShovel-v0")
    env = CustomRewardWrapperRPPO(env)
    env = FlattenObservationWrapper(env)
    env = MultiDiscreteToDictActionWrapper(env)
    return env

# Create a single vectorized environment
vec_env = make_vec_env(make_env, n_envs=1)

# Callback to stop training after fixed steps
class Stop(BaseCallback):
    def __init__(self, max_steps, verbose=0):
        super().__init__(verbose)
        self.max_steps = max_steps

    def _on_step(self) -> bool:
        return self.num_timesteps < self.max_steps

# Save checkpoints during training
checkpoint_callback = CheckpointCallback(
    save_freq=4000,
    save_path="../models/checkpoints/",
    name_prefix="rppo_bc"
)

# Combine stopping and checkpoint callbacks
callback = CallbackList([
    Stop(max_steps=17500),
    checkpoint_callback
])

# Train or load a Recurrent PPO model using BC-based features
def train_rppo_model():
    if os.path.exists(MODEL_PATH + ".zip"):
        print("\nLoading existing RPPO model...\n")
        model = RecurrentPPO.load(MODEL_PATH, env=vec_env, policy_kwargs=policy_kwargs, verbose=1)
    else:
        print("\nStarting new RPPO model...\n")
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
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

# Entry point
def main():
    train_rppo_model()

if __name__ == "__main__":
    main()
