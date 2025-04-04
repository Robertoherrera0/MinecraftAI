import gym
import numpy as np

class DictToMultiDiscreteWrapper(gym.ActionWrapper):
    """
    Turns MineRL's Dict action space into a single MultiDiscrete space
    """
    def __init__(self, env):
        super().__init__(env)
        
        self.action_keys = sorted([k for k in env.action_space.spaces if k != "camera"])
        self.camera_shape = env.action_space.spaces["camera"].shape[0]

        self.multi_discrete_dims = [env.action_space.spaces[k].n for k in self.action_keys]
        self.multi_discrete_dims += [11, 11]  # Discretize camera into -5 to 5 degrees steps

        self.action_space = gym.spaces.MultiDiscrete(self.multi_discrete_dims)

    def action(self, action):
        act = {k: action[i] for i, k in enumerate(self.action_keys)}
        camera_pitch = action[-2] - 5
        camera_yaw = action[-1] - 5
        act["camera"] = np.array([camera_pitch, camera_yaw], dtype=np.float32)
        return act
    