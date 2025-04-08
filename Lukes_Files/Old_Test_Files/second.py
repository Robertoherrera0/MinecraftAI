import gym
import minerl

env = gym.make("MineRLObtainDiamondShovel-v0")
obs = env.reset()

# Print the observation dictionary to see its structure
print(obs)

env.close()

