import gym
import numpy as np

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, debug=False):
        super().__init__(env)
        self.prev_inventory = {}
        self.prev_health = 20
        self.debug = debug

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_inventory = obs.get("inventory", {}).copy()
        self.prev_health = obs.get("life_stats", {}).get("health", 20)
        return obs

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        inventory = obs.get("inventory", {})
        reward = 0

        # Positive rewards
        reward += self._delta("log", inventory, weight=1.0)
        reward += self._delta("sapling", inventory, weight=0.5)
        reward += self._delta("stick", inventory, weight=1.5)

        # Penalize taking damage
        new_health = obs.get("life_stats", {}).get("health", self.prev_health)
        if new_health < self.prev_health:
            reward -= 2.0
        self.prev_health = new_health

        # Penalize looking straight up/down too long based on POV brightness
        reward += self._pov_look_penalty(obs)

        self.prev_inventory = inventory.copy()

        if self.debug:
            print(f"[REWARD] Reward: {reward:.2f}, Inventory: {inventory}")

        return obs, reward, done, info

    def _delta(self, key, inventory, weight=1.0):
        prev = self.prev_inventory.get(key, 0)
        now = inventory.get(key, 0)
        diff = now - prev
        return weight * diff if diff > 0 else 0

    def _pov_look_penalty(self, obs):
        pov = obs.get("pov", None)
        if pov is None:
            return 0

        h = pov.shape[0] // 2
        top_half = pov[:h, :, :]
        bottom_half = pov[h:, :, :]

        top_brightness = top_half.mean()
        bottom_brightness = bottom_half.mean()

        penalty = 0
        if top_brightness > bottom_brightness + 10:
            penalty -= 0.05  # Looking up
        elif bottom_brightness > top_brightness + 10:
            penalty -= 0.05  # Looking down

        return penalty
