# import gym
# import minerl

# # Uncomment to see more logs of the MineRL launch
# # import coloredlogs
# # coloredlogs.install(logging.DEBUG)

# env = gym.make("MineRLObtainDiamondShovel-v0")
# #env = gym.make("MineRLTreechop-v0")
# obs = env.reset()

# # done = False
# # while not done:
# #     ac = env.action_space.noop()
# #     # Spin around to see what is around us
# #     ac["camera"] = [0, 3]
# #     obs, reward, done, info = env.step(ac)
# #     env.render()
# # env.close()

# done = False
# while not done:
#     # Action dictionary (default: no movement)
#     action = env.action_space.noop()
    
#     # Move forward to reach a tree
#     action["forward"] = 1  
#     action["attack"] = 1   # Mine the block in front

#     # Look slightly downwards to aim at the log
#     action["camera"] = [10, 0]  

#     obs, reward, done, info = env.step(action)
#     env.render()

# env.close()


# import gym
# import minerl
# import numpy as np

# def detect_tree(obs):
#     """Detects if a tree is in the agent's view using color filtering."""
#     pov = obs["pov"]  # Get the agent's visual observation

#     # Define RGB range for tree logs (oak logs are brownish)
#     lower_bound = np.array([50, 30, 0])   # Lower RGB threshold
#     upper_bound = np.array([150, 100, 50]) # Upper RGB threshold

#     # Create a mask for pixels within this range
#     tree_mask = ((pov >= lower_bound) & (pov <= upper_bound)).all(axis=-1)

#     # If a significant number of pixels match, a tree is detected
#     return np.sum(tree_mask) > 100  # Adjust threshold as needed

# # Initialize the environment
# env = gym.make("MineRLObtainDiamondShovel-v0")
# obs = env.reset()

# done = False
# found_tree = False

# while not done:
#     action = env.action_space.noop()

#     if not found_tree:
#         # Rotate camera to search for a tree
#         action["camera"] = [0, 3]
#     else:
#         # Move forward and mine if tree is detected
#         action["forward"] = 1  
#         action["attack"] = 1   
#         action["camera"] = [10, 0]  # Look slightly down at logs

#     # Perform action
#     obs, reward, done, info = env.step(action)
#     env.render()

#     # Check if a tree is visible
#     if detect_tree(obs):
#         print("Tree detected! Moving to mine it.")
#         found_tree = True

# env.close()

# import gym
# import minerl
# import numpy as np

# def detect_tree(obs):
#     """Detects if a tree is in the agent's view using color filtering."""
#     pov = obs["pov"]  # Get the agent's visual observation

#     # Narrow RGB range for tree logs (adjusted to reduce false positives)
#     lower_bound = np.array([80, 50, 20])   # More specific brown color
#     upper_bound = np.array([140, 100, 60])

#     # Create a mask for pixels within this range
#     tree_mask = ((pov >= lower_bound) & (pov <= upper_bound)).all(axis=-1)

#     # Threshold: If more than 500 pixels match, assume a tree is in view
#     return np.sum(tree_mask) > 500  

# # Initialize the environment
# env = gym.make("MineRLObtainDiamondShovel-v0")
# obs = env.reset()

# done = False
# found_tree = False
# turning_steps = 0  # Keep track of how many times we've turned

# while not done:
#     action = env.action_space.noop()

#     if not found_tree:
#         # Rotate camera in steps to look for a tree
#         action["camera"] = [0, 10]  
#         turning_steps += 1
        
#         # Stop turning after 36 steps (~360 degrees)
#         if turning_steps > 36:
#             print("No tree found after full rotation.")
#             break  # Exit loop or implement another search strategy
#     else:
#         # Move forward and mine the tree
#         action["forward"] = 1  
#         action["attack"] = 1   
#         action["camera"] = [10, 0]  # Look slightly down at logs

#     # Perform action
#     obs, reward, done, info = env.step(action)
#     env.render()

#     # Check if a tree is visible
#     if detect_tree(obs):
#         print("Tree detected! Moving to mine it.")
#         found_tree = True

# env.close()

import gym
import minerl
import numpy as np

def detect_tree_grid(obs):
    """Detects if a tree is in the agent's nearby grid observation."""
    if "grid" not in obs:
        print("Grid data is not available")
        return False  # Grid data is not available
        

    tree_blocks = {  # Possible tree log types
        "minecraft:oak_log",
        "minecraft:spruce_log",
        "minecraft:birch_log",
        "minecraft:jungle_log",
        "minecraft:acacia_log",
        "minecraft:dark_oak_log"
    }

    # Check if any nearby block is a tree log
    for block in obs["grid"]:
        if block in tree_blocks:
            return True  # Tree detected
    
    return False  # No trees found

# Initialize the environment
env = gym.make("MineRLObtainDiamondShovel-v0")
obs = env.reset()

done = False
found_tree = False

while not done:
    action = env.action_space.noop()

    if not found_tree:
        # Rotate camera to scan for trees
        action["forward"] = 1  
        action["attack"] = 1
        action["jump"] = 1
    else:
        # Move forward and start mining
        action["forward"] = 1  
        action["attack"] = 1   
        action["camera"] = [10, 0]  # Adjust camera down slightly

    # Perform action
    obs, reward, done, info = env.step(action)
    env.render()

    # Check if a tree is nearby in the block grid
    if detect_tree_grid(obs):
        print("Tree detected! Moving to mine it.")
        found_tree = True

env.close()
