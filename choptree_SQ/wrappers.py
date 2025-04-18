# wrappers.py
import gym
import numpy as np

CAMERA_BINS = 21
CAMERA_CENTER = CAMERA_BINS // 2
INVENTORY_KEYS = ["log"]
INPUT_DIM = 64 * 64 * 3 + len(INVENTORY_KEYS)

def flatten_observation(obs):
    import cv2
    pov = obs["pov"][..., :3]
    pov = cv2.resize(pov, (64, 64)).astype(np.uint8)
    inventory = obs.get("inventory", {})
    inv_vec = np.array([inventory.get(k, 0) for k in INVENTORY_KEYS], dtype=np.float32)
    return np.concatenate([pov.flatten(), inv_vec])

class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(INPUT_DIM,), dtype=np.uint8)

    def observation(self, obs):
        return flatten_observation(obs)

class MultiDiscreteToDictActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_keys = ["attack", "back", "forward", "jump", "left", "right"]
        self.action_space = gym.spaces.MultiDiscrete([2] * 6 + [CAMERA_BINS, CAMERA_BINS])

    def action(self, action):
        act = {k: int(action[i]) for i, k in enumerate(self.action_keys)}
        yaw_bin, pitch_bin = action[-2], action[-1]
        act["camera"] = np.array([pitch_bin - CAMERA_CENTER, yaw_bin - CAMERA_CENTER], dtype=np.float32)
        return act
