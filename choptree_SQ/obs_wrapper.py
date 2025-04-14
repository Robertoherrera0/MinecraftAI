import gym
import cv2
import numpy as np
from gym import spaces

class FlattenObservationWrapper(gym.ObservationWrapper):
    """
    Observation wrapper that:
      - Resizes the 'pov' image to (64, 64, 3)
      - Flattens the entire inventory into a consistent 1D vector
    """

    def __init__(self, env, resize=(64, 64)):
        super().__init__(env)
        self.resize = resize

        # Get the full set of inventory keys from the unwrapped env
        dummy_obs = env.reset()
        self.inv_items = sorted(dummy_obs.get("inventory", {}).keys())

        # Define new observation space
        pov_shape = (resize[0], resize[1], 3)
        self.pov_space = spaces.Box(low=0, high=255, shape=pov_shape, dtype=np.uint8)
        self.inv_space = spaces.Box(low=0, high=9999, shape=(len(self.inv_items),), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "pov": self.pov_space,
            "inv": self.inv_space
        })

    def observation(self, obs):
        # Resize POV
        pov = obs["pov"]
        if pov.shape[-1] > 3:
            pov = pov[..., :3]
        pov = cv2.resize(pov, (self.resize[1], self.resize[0]))
        pov = pov.astype(np.uint8)

        # Flatten full inventory using consistent order
        inventory_dict = obs.get("inventory", {})
        inventory_vec = [inventory_dict.get(item, 0) for item in self.inv_items]
        inv_array = np.array(inventory_vec, dtype=np.float32)

        return {
            "pov": pov,
            "inv": inv_array
        }
