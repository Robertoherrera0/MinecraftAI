import gym
import numpy as np

class LogRewardWrapper(gym.Wrapper):
    def __init__(self, env, debug=False):
        super().__init__(env)
        self.prev_logs = 0
        self.debug = debug

    def reset(self, **kwargs):
        # Get the first observation
        obs = self.env.reset(**kwargs)

        # If observation is a flattened array, unflatten it
        obs = self.unflatten_obs(obs)
        
        # Initialize previous logs count
        self.prev_logs = self.count_logs(obs["inventory"])  # Access the "inventory" from unflattened obs
        return obs

    def step(self, action):
        # Take a step in the environment
        obs, _, done, info = self.env.step(action)

        # If observation is a flattened array, unflatten it
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
        # Since inventory is a flat array, we will count the logs by checking values
        # Assumes the inventory contains the "log" item at a specific index, e.g., index 0.
        return inventory[0]  # Replace with correct index for logs if needed

    def unflatten_obs(self, obs):
        # Unflatten the observation back to its original structure
        pov_size = 64 * 64 * 3  # Adjust based on actual size of the image
        pov = obs[:pov_size].reshape(64, 64, 3)  # Reshape to the original image shape (64x64x3)
        inventory = obs[pov_size:]  # The rest is inventory data

        # Return the unflattened observation structure
        return {"pov": pov, "inventory": inventory}

    def flatten_obs(self, obs):
        # Flatten the observation (POV and inventory) back into a single array
        pov_flat = obs["pov"].flatten()  # Flatten the POV image
        inventory_flat = obs["inventory"].flatten()  # Flatten the inventory
        return np.concatenate([pov_flat, inventory_flat])  # Combine into a single array
