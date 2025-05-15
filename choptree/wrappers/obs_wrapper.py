import gym
import cv2
import numpy as np
from gym import spaces

# Flattens MineRL observations (POV + inventory) into a single array
class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, resize=(64, 64)):
        super().__init__(env)
        self.resize = resize

        # Use initial obs to detect inventory keys
        dummy_obs = env.reset()
        self.inv_items = sorted(dummy_obs.get("inventory", {}).keys())

        # Define observation space after flattening
        pov_shape = (resize[0], resize[1], 3)
        self.pov_space = spaces.Box(low=0, high=255, shape=pov_shape, dtype=np.uint8)
        self.inv_space = spaces.Box(low=0, high=9999, shape=(len(self.inv_items),), dtype=np.float32)

        total_obs_size = resize[0] * resize[1] * 3 + len(self.inv_items)
        self.observation_space = spaces.Box(low=0, high=255, shape=(total_obs_size,), dtype=np.uint8)

    def observation(self, obs):
        pov = obs["pov"]
        if pov.shape[-1] > 3:
            pov = pov[..., :3]  # Drop alpha if present
        pov = cv2.resize(pov, (self.resize[1], self.resize[0]))
        pov = pov.astype(np.uint8)

        inventory_dict = obs.get("inventory", {})
        inventory_vec = [inventory_dict.get(item, 0) for item in self.inv_items]
        inv_array = np.array(inventory_vec, dtype=np.float32)

        # Return as a single numpy array instead of a dictionary
        return np.concatenate([pov.flatten(), inv_array])  # Flatten pov and inventory and concatenate
