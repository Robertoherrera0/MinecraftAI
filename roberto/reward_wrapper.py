import gym
import numpy as np

class LogRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_logs = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_logs = self.count_logs(obs.get("inventory", {}))
        return obs

    def step(self, action):
        obs, env_reward, done, info = self.env.step(action)
        current_logs = self.count_logs(obs.get("inventory", {}))
        # Additional shaped reward for net logs gained
        gained = current_logs - self.prev_logs
        self.prev_logs = current_logs

        # Combine environment's reward with shaped reward
        shaped_reward = env_reward + gained
        return obs, shaped_reward, done, info

    def count_logs(self, inventory):
        return inventory.get("log", 0)
