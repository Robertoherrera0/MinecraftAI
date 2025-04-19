import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from custom_reward_wrapper import CustomRewardWrapper
from wrappers import FlattenObservationWrapper, MultiDiscreteToDictActionWrapper
import minerl

CHECKPOINT_PATH = "checkpoints/ppo_16000_steps" 

def make_env():
    env = gym.make("MineRLObtainDiamondShovel-v0")
    env = CustomRewardWrapper(env)
    env = FlattenObservationWrapper(env)
    env = MultiDiscreteToDictActionWrapper(env)
    return env

def main():
    print("Loading environment and PPO model...")
    env = make_env()
    model = PPO.load(CHECKPOINT_PATH)

    # get raw env for render
    raw_env = env.envs[0] if hasattr(env, "envs") else env

    obs = env.reset()
    total_reward = 0
    rewards = []
    yaw_bins = []
    pitch_bins = []
    step = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

        raw_env.render()  

        total_reward += reward
        rewards.append(total_reward)
        pitch_bins.append(action[-2])  # pitch
        yaw_bins.append(action[-1])    # yaw

        print(f"[STEP {step}] Reward: {reward:.2f} | Pitch: {pitch_bins[-1]} | Yaw: {yaw_bins[-1]}")
        step += 1

        if done or step > 3000: # delete "step > 3000" if you want to run longer episodes"
            break

    env.close()
    print(f"Evaluation done | Steps: {step} | Total reward: {total_reward}")

    # --- Plotting ---
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
