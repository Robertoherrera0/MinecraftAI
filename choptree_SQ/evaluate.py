import gym
import minerl
import torch
import numpy as np
from stable_baselines3 import PPO

from obs_wrapper import FlattenObservationWrapper
from action_wrapper import DictToMultiDiscreteWrapper
from reward_wrapper import LogRewardWrapper

def make_env(debug=False):
    env = gym.make("MineRLObtainDiamondShovel-v0")
    env = DictToMultiDiscreteWrapper(env)
    env = FlattenObservationWrapper(env, inv_items=["log"])
    env = LogRewardWrapper(env, debug=debug)
    return env

def load_bc_model(model_path):
    # Load the behavior cloning model
    model = torch.load(model_path)  # Load the trained model from the path
    model.eval()  # Set the model to evaluation mode
    return model

def main():
    print("Loading environment and model...")
    env = make_env(debug=False)

    # Load the pre-trained behavior cloning model
    bc_model = load_bc_model("/models/bc_model.pth")  # Adjust the path to where the model is saved

    obs = env.reset()
    max_logs = 0

    print("Starting rendering...")
    for step in range(1000):
        # Get the action from the behavior cloning model
        observation_tensor = torch.tensor(obs['pov'].flatten(), dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        action = bc_model(observation_tensor).argmax(dim=-1).item()  # Get the action with max probability

        obs, reward, done, _ = env.step(action)
        env.render()

        logs = obs["inv"][0]
        max_logs = max(max_logs, logs)

        print(f"[STEP {step}] Reward: {reward:.2f} | Logs: {logs}")

        if done:
            print("Done. Resetting environment...\n")
            obs = env.reset()

    print(f"\nMax logs collected: {max_logs}")
    env.close()

if __name__ == "__main__":
    main()
