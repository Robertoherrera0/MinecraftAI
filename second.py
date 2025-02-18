import gym
import minerl
import numpy as np

env = gym.make("MineRLObtainDiamondShovel-v0")  # Change to your environment
obs = env.reset()

# Print the shape of the grid
if "grid" in obs:
    print("Grid shape:", np.shape(obs["grid"]))  # Expected: (5, 5, 5)
else:
    print("No grid data found.")

env.close()
