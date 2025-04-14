import gym
import numpy as np

class DictToMultiDiscreteWrapper(gym.ActionWrapper):
    """
    Converts MineRL's Dict action space into a MultiDiscrete space.
    Keeps only relevant actions (movement, jump, attack) and discretizes camera (yaw + pitch).
    """

    def __init__(self, env):
        super().__init__(env)

        # Actions to keep (excluding crafting, equip, etc.)
        self.action_keys = sorted([
            k for k in env.action_space.spaces
            if k not in ["camera"]
        ])

        # Discretization for camera (yaw and pitch): from -10 to 10 degrees
        self.camera_bins = 21
        self.yaw_offset = self.pitch_offset = self.camera_bins // 2

        # Build dimensions for MultiDiscrete
        self.multi_discrete_dims = [env.action_space.spaces[k].n for k in self.action_keys]
        self.multi_discrete_dims.extend([self.camera_bins, self.camera_bins])  # yaw, pitch

        self.action_space = gym.spaces.MultiDiscrete(self.multi_discrete_dims)

    def action(self, action):
        # Map MultiDiscrete back to Dict format
        act = {k: action[i] for i, k in enumerate(self.action_keys)}

        # Extract yaw and pitch (last two)
        camera_yaw = action[-2] - self.yaw_offset
        camera_pitch = action[-1] - self.pitch_offset
        act["camera"] = np.array([float(camera_pitch), float(camera_yaw)], dtype=np.float32)

        return act
