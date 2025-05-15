import gym
import numpy as np

# CURRENTLY NOT USING, WE ARE USING CUSTOM REWARD WRAPPER 
class LogRewardWrapper(gym.Wrapper):
    def __init__(self, env, debug=False):
        super().__init__(env)
        self.prev_logs = 0
        self.debug = debug

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_logs = self.count_logs(obs["inventory"])
        return self.flatten_obs(obs)

    def step(self, action):
        obs, _, done, info = self.env.step(action)

        obs = self.unflatten_obs(obs)
        
        # Calculate the reward based on logs collected
        current_logs = self.count_logs(obs["inventory"])
        gained = current_logs - self.prev_logs
        self.prev_logs = current_logs
        
        # Reward is based solely on logs gained
        shaped_reward = 0.5 * gained  # Adjust reward shaping if needed

        if self.debug:
            print(f"[REWARD] Logs: {current_logs}, Gained: {gained}, Reward: {shaped_reward}")

        return self.flatten_obs(obs), shaped_reward, done, info  # Return flattened observation

    def count_logs(self, inventory):
        return inventory[0]  # Replace with correct index for logs if needed

    def unflatten_obs(self, obs):
        pov_size = 64 * 64 * 3  # Adjust based on actual size of the image
        pov = obs[:pov_size].reshape(64, 64, 3)  # Reshape to the original image shape (64x64x3)
        inventory = obs[pov_size:]  

        return {"pov": pov, "inventory": inventory}

    def flatten_obs(self, obs):
        # Flatten the observation (POV and inventory) back into a single array
        pov_flat = obs["pov"].flatten()  # Flatten the POV image
        inventory_flat = obs["inventory"].flatten()  # Flatten the inventory
        return np.concatenate([pov_flat, inventory_flat])  # Combine into a single array
