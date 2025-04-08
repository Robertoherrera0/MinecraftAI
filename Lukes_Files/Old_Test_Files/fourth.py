import gym
import minerl
import numpy as np
import time

# Define the RGB values for each wood type
WOOD_COLORS = {
    "wood_1": np.array([166, 157, 111]),  # #a69d6f
    "wood_2": np.array([145, 117, 77]),   # #91754d
    "wood_3": np.array([99, 73, 43]),     # #63492b
    "wood_4": np.array([54, 35, 16]),     # #362310
    "wood_5": np.array([142, 104, 79])    # #8e684f
}

# Function to check if a pixel's color is within a tolerance range
def is_similar_color(pixel, target_color, tolerance=3):
    return np.all(np.abs(pixel - target_color) <= tolerance)

# Initialize the environment
env = gym.make("MineRLObtainDiamondShovel-v0")  # Modify to your environment
obs = env.reset()

# Function to move the agent in the given direction (forward, backward, left, right)
def move_agent(action):
    action_dict = {
        'forward': [1, 0, 0],  # Move forward
        'backward': [-1, 0, 0], # Move backward
        'left': [0, 1, 0],      # Move left
        'right': [0, -1, 0],    # Move right
    }
    return action_dict.get(action, [0, 0, 0])

# Look through the field of view (pov) to find matching wood blocks
wood_positions = []

# Main loop to continuously walk towards the detected wood
while True:
    obs = env.reset()  # Reset the environment if needed
    
    # Get the image of the agent's view
    pov_image = obs['pov']  # pov is typically a 3D array (height, width, channels)
    height, width, _ = pov_image.shape
    wood_found = False  # Flag to track if wood is found in the field of view

    # Search for wood colors in the environment's field of view
    for i in range(height):
        for j in range(width):
            pixel = pov_image[i, j]
            for wood, target_color in WOOD_COLORS.items():
                if is_similar_color(pixel, target_color):
                    wood_positions.append((i, j))
                    wood_found = True

    # If we found wood, move towards it
    if wood_found:
        # For simplicity, choose the first detected wood position
        target_position = wood_positions[0]  # (i, j)

        # Calculate a basic movement strategy (move towards the target position)
        target_x, target_y = target_position
        print(f"Detected wood at position: {target_position}")

        # Simple approach: move forward or backward depending on position
        if target_x < height // 2:
            print("Moving forward")
            action = move_agent('forward')
        elif target_x > height // 2:
            print("Moving backward")
            action = move_agent('backward')
        else:
            action = [0, 0, 0]  # No movement needed if wood is centered

        # Execute action
        obs, reward, done, info = env.step(action)
        
        # Render the environment after taking the step
        env.render()

        # Add a small delay for smoother action
        time.sleep(0.1)

        # Check if episode is done (e.g., agent reaches the goal or some condition)
        if done:
            print("Episode finished.")
            break
    else:
        print("No wood detected. Continuing to search.")
        # Continue to search until wood is found
        obs, reward, done, info = env.step([0, 0, 0])  # Stay in place if no wood detected
        env.render()
        time.sleep(0.1)
