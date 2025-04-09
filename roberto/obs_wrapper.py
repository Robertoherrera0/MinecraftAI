import gym
import cv2
import numpy as np
from gym import spaces

class FlattenObservationWrapper(gym.ObservationWrapper):
    """
    new obs space
      - 'pov': resized image (64x64x3)
      - 'inventory': flattened into a 1D vector of selected items

    Discards or ignores anything else (like 'equipped_items', 'location_stats', etc.).
    The resulting observation is a Dict with exactly two keys: {'pov', 'inv'},
    neither of which is a nested dict. This satisfies SB3's requirements.
    """

    def __init__(self, env, resize=(64,64), inv_items=None):
        super().__init__(env)
        self.resize = resize
        if inv_items is None:
            # track logs
            inv_items = ["log"]  
        self.inv_items = inv_items

        # 1) POV shape after resizing
        pov_shape = (resize[0], resize[1], 3)
        self.pov_space = spaces.Box(low=0, high=255, shape=pov_shape, dtype=np.uint8)

        # 2) Inventory shape is # of items you track
        self.inv_space = spaces.Box(low=0, high=9999, shape=(len(self.inv_items),), dtype=np.float32)

        # Combine into top-level dict
        self.observation_space = spaces.Dict({
            "pov": self.pov_space,
            "inv": self.inv_space
        })

    def observation(self, obs):
        pov = obs["pov"]
        if pov.shape[-1] > 3:
            pov = pov[..., :3]
        pov = cv2.resize(pov, (self.resize[1], self.resize[0]))
        pov = pov.astype(np.uint8)

        # Flatten the selected inventory items
        inventory_vec = []
        inventory_dict = obs.get("inventory", {})
        for item in self.inv_items:
            val = inventory_dict.get(item, 0)
            inventory_vec.append(val)
        inv_array = np.array(inventory_vec, dtype=np.float32)

        return {
            "pov": pov,
            "inv": inv_array
        }
