import gym

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
        
        shaped_reward = 1.0 * gained

        if self.debug:
            print(f"[REWARD] Logs: {current_logs}, Gained: {gained}, Reward: {shaped_reward}")

        return obs, shaped_reward, done, info

    def count_logs(self, inventory):
        return sum(v for k, v in inventory.items() if "log" in k.lower())
