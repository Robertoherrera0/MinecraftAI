import gym
import numpy as np

# Dummy MineRL-like environment with Dict action space
class FakeMineRLEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Dict({
            "forward": gym.spaces.Discrete(2),
            "back": gym.spaces.Discrete(2),
            "jump": gym.spaces.Discrete(2),
            "attack": gym.spaces.Discrete(2),
            "camera": gym.spaces.Box(low=-180, high=180, shape=(2,), dtype=np.float32)
        })

    def step(self, action):
        return None, 0.0, False, {}

    def reset(self):
        return {}

# Your wrapper
class DictToMultiDiscreteWrapper(gym.ActionWrapper):
    """
    Converts MineRL's Dict action space into a MultiDiscrete space.
    Keeps only relevant actions (movement, jump, attack) and discretizes camera (yaw + pitch).
    """

    def __init__(self, env):
        super().__init__(env)

        self.action_keys = sorted([
            k for k in env.action_space.spaces
            if k not in ["camera"]
        ])

        self.camera_bins = 21
        self.yaw_offset = self.pitch_offset = self.camera_bins // 2

        self.multi_discrete_dims = [env.action_space.spaces[k].n for k in self.action_keys]
        self.multi_discrete_dims.extend([self.camera_bins, self.camera_bins])  # yaw, pitch

        self.action_space = gym.spaces.MultiDiscrete(self.multi_discrete_dims)

    def action(self, action):
        act = {k: action[i] for i, k in enumerate(self.action_keys)}

        camera_yaw = action[-2] - self.yaw_offset
        camera_pitch = action[-1] - self.pitch_offset
        act["camera"] = np.array([float(camera_pitch), float(camera_yaw)], dtype=np.float32)

        return act

# --- Test ---
env = DictToMultiDiscreteWrapper(FakeMineRLEnv())

sample_action = env.action_space.sample()
converted = env.action(sample_action)

print("Original MultiDiscrete:", sample_action)
print("Converted Dict Action:", converted)
