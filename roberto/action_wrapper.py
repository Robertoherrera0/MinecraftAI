import gym
import numpy as np

class DictToMultiDiscreteWrapper(gym.ActionWrapper):
    """
    Converts MineRL's Dict action space into a MultiDiscrete space for SB3.
    Keeps only relevant actions, including camera yaw (horizontal look).
    """

    def __init__(self, env):
        super().__init__(env)

        # Actions to keep (excluding camera pitch, crafting, etc.)
        self.action_keys = sorted([
            k for k in env.action_space.spaces
            if k not in ["camera"]
        ])
        
        # Build dimensions for MultiDiscrete action space
        self.multi_discrete_dims = [env.action_space.spaces[k].n for k in self.action_keys]

        # Add camera yaw only: -5 to 5 → 11 discrete options
        self.camera_bins = 11  # from -5 to 5 degrees
        self.multi_discrete_dims.append(self.camera_bins)

        self.action_space = gym.spaces.MultiDiscrete(self.multi_discrete_dims)

    def action(self, action):
        # Map MultiDiscrete back to Dict format
        act = {k: action[i] for i, k in enumerate(self.action_keys)}

        # Camera yaw = last element, pitch = always 0
        camera_yaw = action[-1] - (self.camera_bins // 2)  # -5 to 5
        act["camera"] = np.array([0.0, float(camera_yaw)], dtype=np.float32)
        
        return act
