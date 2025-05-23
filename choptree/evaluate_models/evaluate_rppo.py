import gym
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO  # type:ignore
import minerl  # type:ignore
import sys

from wrappers.custom_reward_wrapper import CustomRewardWrapperRPPO
from wrappers.wrappers import FlattenObservationWrapper, MultiDiscreteToDictActionWrapper
from features.bc_extractor import BCFeatureExtractor

# MODEL_PATH = "checkpoints/rppo_bc_12000_steps"
MODEL_PATH = "models/rppo_bc_model"

# Set up the MineRL environment with custom wrappers
def make_env():
    env = gym.make("MineRLObtainDiamondShovel-v0")
    env = CustomRewardWrapperRPPO(env)
    env = FlattenObservationWrapper(env)
    env = MultiDiscreteToDictActionWrapper(env)
    return env

# Run the evaluation loop
def main():
    print("Loading environment and Recurrent PPO model...")
    env = make_env()
    model = RecurrentPPO.load(MODEL_PATH)

    raw_env = env.envs[0] if hasattr(env, "envs") else env

    obs = env.reset()
    total_reward = 0
    rewards = []
    yaw_bins = []
    pitch_bins = []
    step = 0

    lstm_states = None  # Recurrent PPO needs to track hidden state
    episode_starts = np.ones((1,), dtype=bool)  # resets LSTM at start

    while True:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts)
        obs, reward, done, _ = env.step(action)

        raw_env.render()

        episode_starts = np.array([done])
        total_reward += reward
        rewards.append(total_reward)
        pitch_bins.append(action[-2])  # pitch
        yaw_bins.append(action[-1])    # yaw
        print("Full action:", action)
        print(f"[STEP {step}] Reward: {reward:.5f} | Pitch: {pitch_bins[-1]} | Yaw: {yaw_bins[-1]}")
        step += 1

        if done or step > 15_000:
            break

    env.close()
    print(f"Evaluation done | Steps: {step} | Total reward: {total_reward}")

    # Plot rewards and camera movement
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
