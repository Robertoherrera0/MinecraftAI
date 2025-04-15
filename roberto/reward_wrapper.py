import gym
import numpy as np

class LogRewardWrapper(gym.Wrapper):
    def __init__(self, env, debug=False):
        super().__init__(env)
        self.prev_logs = 0
        self.debug = debug

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_logs = self.count_logs(obs.get("inventory", {}))
        return obs

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        current_logs = self.count_logs(obs.get("inventory", {}))

        gained = current_logs - self.prev_logs
        self.prev_logs = current_logs

        attack_pressed = action[2] == 1
        shaped_reward = 0.5 * gained + (0.05 if attack_pressed else 0)

        if self.is_looking_at_tree(obs["pov"]):
            shaped_reward += 0.1  # Reward for looking at a tree

        if self.debug:
            print(f"[REWARD] Logs: {current_logs}, Gained: {gained}, Looking at tree: {self.is_looking_at_tree(obs['pov'])}, Reward: {shaped_reward}")

        return obs, shaped_reward, done, info

    def count_logs(self, inventory):
        return inventory.get("log", 0)

    def is_looking_at_tree(self, pov):
        """
        Check if the center region of the screen contains green or brown colors
        commonly associated with leaves/logs.
        """
        h, w, _ = pov.shape
        center = pov[h//2 - 4:h//2 + 4, w//2 - 4:w//2 + 4]  # 8x8 center patch
        avg_color = np.mean(center, axis=(0, 1))  # RGB mean

        # Rough RGB thresholds for green (leaves) or brown (logs)
        is_greenish = avg_color[1] > 100 and avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]
        is_brownish = avg_color[0] > 60 and avg_color[1] > 40 and avg_color[2] < 60 and avg_color[0] > avg_color[1]

        return is_greenish or is_brownish

