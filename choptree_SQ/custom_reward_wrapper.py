import gym

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

        # Reward collecting logs
        reward += self._delta("log", inventory, weight=1.0)
        # Reward collecting saplings
        reward += self._delta("sapling", inventory, weight=0.5)
        # Reward for crafting sticks
        reward += self._delta("stick", inventory, weight=1.5)
        # Penalize damage taken (if available)
        new_health = obs.get("life_stats", {}).get("health", self.prev_health)
        if new_health < self.prev_health:
            reward -= 2.0
        self.prev_health = new_health

        self.prev_inventory = inventory.copy()

        if self.debug:
            print(f"[REWARD] Reward: {reward:.2f}, Inventory: {inventory}")

        return obs, reward, done, info

    def _delta(self, key, inventory, weight=1.0):
        prev = self.prev_inventory.get(key, 0)
        now = inventory.get(key, 0)
        diff = now - prev
        return weight * diff if diff > 0 else 0
