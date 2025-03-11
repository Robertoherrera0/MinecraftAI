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
    # # Reset the environment with a custom starting inventory
    # obs = env.reset()
obs['inventory']['iron_pickaxe'] = 1  # Start with an iron pickaxe

done = False
found_tree = False

while not done:
    action = env.action_space.noop()
    action["inventory"] = 1
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