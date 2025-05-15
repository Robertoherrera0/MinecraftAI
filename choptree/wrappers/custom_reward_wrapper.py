import gym
import numpy as np
import gym.spaces
import cv2

"""
in controller.py replace this 

from reward_wrapper import LogRewardWrapper
...
env = LogRewardWrapper(base_env)


with this

from reward_wrapper import CustomRewardWrapper
...
env = CustomRewardWrapper(base_env, debug=True)
"""

# Basic custom reward wrapper for debugging/log collection
class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, debug=False):
        super().__init__(env)
        self.prev_inventory = {}
        self.prev_health = 20  # max in MineRL
        self.debug = debug

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_inventory = obs.get("inventory", {}).copy()
        self.prev_health = obs.get("equipped_items", {}).get("mainhand", {}).get("damage", 0)
        return obs

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        inventory = obs.get("inventory", {})
        reward = 0

        # Inventory-based rewards
        reward += self._delta("log", inventory, weight=1.0)
        reward += self._delta("sapling", inventory, weight=0.5)
        reward += self._delta("stick", inventory, weight=1.5)

        # Penalize damage taken (if available)
        new_health = obs.get("life_stats", {}).get("health", self.prev_health)
        if new_health < self.prev_health:
            reward -= 2.0
        self.prev_health = new_health

        self.prev_inventory = inventory.copy()

        if self.debug:
            print(f"Reward: {reward:.2f}")

        return obs, reward, done, info

    # Track increase in inventory items
    def _delta(self, key, inventory, weight=1.0):
        prev = self.prev_inventory.get(key, 0)
        now = inventory.get(key, 0)
        diff = now - prev
        return weight * diff if diff > 0 else 0


# RPPO-specific reward shaping with camera-based feedback
class CustomRewardWrapperRPPO(gym.Wrapper):
    def __init__(self, env, debug=False):
        super().__init__(env)
        self.prev_inventory = {}
        self.prev_health = 20 
        self.prev_camera = np.array([0.0, 0.0])
        self.debug = debug

        self.tree_items = {
            "log": 1.0,
            "stick": 1.5,
            "sapling": 0.5,
            "oak_log": 1.0,
            "birch_log": 1.0,
            "jungle_log": 1.0,
            "acacia_log": 1.0,
            "dark_oak_log": 1.0,
        }

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_inventory = obs.get("inventory", {}).copy()
        self.prev_health = obs.get("life_stats", {}).get("health", 20)
        self.prev_camera = np.array([0.0, 0.0])
        return obs

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        reward = 0
        inventory = obs.get("inventory", {})
        camera = action.get("camera", np.array([0.0, 0.0]))
        pitch_delta = abs(camera[0])
        yaw_delta = abs(camera[1])

        # Reward for collecting tree-related items
        for item, weight in self.tree_items.items():
            reward += self._delta(item, inventory, weight=weight)

        # Reward camera movement (encourages looking around)
        if not np.allclose(camera, self.prev_camera, atol=0.01):
            reward += 0.0001 * (pitch_delta + yaw_delta)
        else:
            reward -= 0.001  # small penalty for twitching the same way

        self.prev_camera = camera.copy()

        # small reward for keeping camera centered
        pitch_bin = int(camera[0] + 10)
        yaw_bin = int(camera[1] + 10)
        if 7 <= pitch_bin <= 14 and 7 <= yaw_bin <= 14:
            reward += 0.02

        # Penalize pressing opposite movement keys
        if action.get("forward", 0) and action.get("back", 0):
            reward -= 0.1
        if action.get("left", 0) and action.get("right", 0):
            reward -= 0.1

        # Penalize health loss
        new_health = obs.get("life_stats", {}).get("health", self.prev_health)
        if new_health < self.prev_health:
            reward -= 2.0
        self.prev_health = new_health

        self.prev_inventory = inventory.copy()

        if self.debug:
            print(f"Reward: {reward:.2f}  | Camera: {camera}")

        return obs, reward, done, info

    # Track increase in tree-related inventory items
    def _delta(self, key, inventory, weight=1.0):
        prev = self.prev_inventory.get(key, 0)
        now = inventory.get(key, 0)
        diff = now - prev
        return weight * diff if diff > 0 else 0


# CNN-compatible reward wrapper with resized image and simplified obs
class CustomRewardWrapperCNN(gym.Wrapper):
    def __init__(self, env, debug=False):
        super().__init__(env)
        self.debug = debug
        self.prev_inventory = {}
        self.prev_health = 20
        self.prev_camera = np.array([0.0, 0.0])
        self.observation_space = gym.spaces.Dict({
            "pov": gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "inv": gym.spaces.Box(low=0.0, high=2304.0, shape=(1,), dtype=np.float32)
        })

        self.tree_items = {
            "log": 1.0,
            "stick": 1.5,
            "sapling": 0.5,
            "oak_log": 1.0,
            "birch_log": 1.0,
            "jungle_log": 1.0,
            "acacia_log": 1.0,
            "dark_oak_log": 1.0,
        }

    def reset(self, **kwargs):
        raw_obs = self.env.reset(**kwargs)
        self.prev_inventory = raw_obs.get("inventory", {}).copy()
        self.prev_health = raw_obs.get("life_stats", {}).get("health", 20)
        self.prev_camera = np.array([0.0, 0.0])
        return self._convert_obs(raw_obs)

    def step(self, action):
        raw_obs, _, done, info = self.env.step(action)
        reward = 0

        inventory = raw_obs.get("inventory", {})
        camera = action.get("camera", np.array([0.0, 0.0]))
        pitch_delta = abs(camera[0])
        yaw_delta = abs(camera[1])

        # Reward for collecting items
        for item, weight in self.tree_items.items():
            reward += self._delta(item, inventory, weight)

        # Reward camera movement
        if not np.allclose(camera, self.prev_camera, atol=0.01):
            reward += 0.0001 * (pitch_delta + yaw_delta)
        else:
            reward -= 0.001

        self.prev_camera = camera.copy()

        # # Encourage centered camera
        # pitch_bin = int(camera[0] + 10)
        # yaw_bin = int(camera[1] + 10)
        # if 5 <= pitch_bin <= 15 and 5 <= yaw_bin <= 15:
        #     reward += 0.005

        # Penalize opposite key presses
        if action.get("forward", 0) and action.get("back", 0):
            reward -= 0.1
        if action.get("left", 0) and action.get("right", 0):
            reward -= 0.1

        # Penalize health loss
        new_health = raw_obs.get("life_stats", {}).get("health", self.prev_health)
        if new_health < self.prev_health:
            reward -= 2.0
        self.prev_health = new_health

        self.prev_inventory = inventory.copy()

        if self.debug:
            print(f"Reward: {reward:.3f} | Camera: {camera}")

        return self._convert_obs(raw_obs), reward, done, info

    # Track increase in tree items
    def _delta(self, key, inventory, weight=1.0):
        prev = self.prev_inventory.get(key, 0)
        now = inventory.get(key, 0)
        diff = now - prev
        return weight * diff if diff > 0 else 0

    # Resize image and build simplified observation dict
    def _convert_obs(self, raw_obs):
        pov = raw_obs["pov"][..., :3]  # RGB only
        pov = cv2.resize(pov, (64, 64), interpolation=cv2.INTER_AREA).astype(np.uint8)  # <-- RESIZE HERE
        inv = raw_obs.get("inventory", {})
        inv_vec = np.array([inv.get("log", 0)], dtype=np.float32)
        return {
            "pov": pov,
            "inv": inv_vec
        }
