import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO  # type: ignore
import minerl  # type: ignore
import sys

# Fix import path for custom feature extractor (if used)
sys.modules['bc_extractor'] = sys.modules['features.bc_extractor']

from wrappers.custom_reward_wrapper import CustomRewardWrapper
from wrappers.wrappers import FlattenObservationWrapper, MultiDiscreteToDictActionWrapper

# Path to trained PPO model
# MODEL_PATH = "checkpoints/ppo_bc_8000_steps"
MODEL_PATH = "models/ppo_bc_model"

# Create the MineRL environment with custom wrappers
def make_env():
    env = gym.make("MineRLObtainDiamondShovel-v0")
    env = CustomRewardWrapper(env)
    env = FlattenObservationWrapper(env)
    env = MultiDiscreteToDictActionWrapper(env)
    return env

# Run evaluation loop
def main():
    print("Loading environment and PPO model...")
    env = make_env()
    model = PPO.load(MODEL_PATH)

    # Needed for rendering
    raw_env = env.envs[0] if hasattr(env, "envs") else env

    obs = env.reset()
    total_reward = 0
    rewards = []
    yaw_bins = []
    pitch_bins = []
    step = 0

    while True:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

        raw_env.render()  # show the game window

        total_reward += reward
        rewards.append(total_reward)
        pitch_bins.append(action[-2])  # camera pitch
        yaw_bins.append(action[-1])    # camera yaw

        print(f"[STEP {step}] Reward: {reward:.5f} | Pitch: {pitch_bins[-1]} | Yaw: {yaw_bins[-1]}")
        step += 1

        # Stop after 3000 steps or when episode ends
        if done or step > 3000:
            break

    env.close()
    print(f"Evaluation done | Steps: {step} | Total reward: {total_reward}")

    # Plot total reward and camera movements
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Cumulative Reward")
    plt.xlabel("Step")
    plt.ylabel("Total Reward")

    plt.subplot(1, 2, 2)
    plt.plot(yaw_bins, label="Yaw")
    plt.plot(pitch_bins, label="Pitch")
    plt.title("Camera Movement (Yaw / Pitch)")
    plt.xlabel("Step")
    plt.ylabel("Camera Bin")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
