import gym
import minerl
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from obs_wrapper import FlattenObservationWrapper
from action_wrapper import DictToMultiDiscreteWrapper
from reward_wrapper import LogRewardWrapper

DEBUG_MODE = True  # Set to False to silence reward printouts

def make_env():
    env = gym.make("MineRLObtainDiamondShovel-v0")

    # Apply wrappers
    env = DictToMultiDiscreteWrapper(env)
    env = FlattenObservationWrapper(env, inv_items=["log"])
    env = LogRewardWrapper(env)

    return env

def train():
    env = DummyVecEnv([make_env])

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_choptree_tensorboard"
    )

    print("[INFO] Starting training...")
    model.learn(total_timesteps=10000)
    model.save("ppo_choptree_model")

    env.close()

if __name__ == "__main__":
    train()
