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
def is_similar_color(pixel, target_color, tolerance=30):
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

# Limit the search to the central area of the FOV (smaller region to speed up the search)
def search_for_wood_in_fov(pov_image):
    height, width, _ = pov_image.shape
    center_x, center_y = height // 2, width // 2

    # Define a small search area around the center (for faster processing)
    search_radius = 50  # Pixels in each direction to limit the search
    top_left_x = max(center_x - search_radius, 0)
    bottom_right_x = min(center_x + search_radius, height)
    top_left_y = max(center_y - search_radius, 0)
    bottom_right_y = min(center_y + search_radius, width)

    wood_positions = []
    for i in range(top_left_x, bottom_right_x):
        for j in range(top_left_y, bottom_right_y):
            pixel = pov_image[i, j]
            for wood, target_color in WOOD_COLORS.items():
                if is_similar_color(pixel, target_color):
                    wood_positions.append((i, j))

    return wood_positions

# Main loop to continuously walk towards the detected wood
while True:
    obs = env.reset()  # Reset the environment if needed
    
    # Get the image of the agent's view (only render as RGB array for performance)
    pov_image = obs['pov']  # pov is typically a 3D array (height, width, channels)
    
    # Search for wood colors in the smaller field of view
    wood_positions = search_for_wood_in_fov(pov_image)

    if wood_positions:
        # For simplicity, choose the first detected wood position
        target_position = wood_positions[0]  # (i, j)
        target_x, target_y = target_position
        print(f"Detected wood at position: {target_position}")

        # Simple approach: move forward or backward depending on position
        if target_x < pov_image.shape[0] // 2:
            print("Moving forward")
            action = move_agent('forward')
        elif target_x > pov_image.shape[0] // 2:
            print("Moving backward")
            action = move_agent('backward')
        else:
            action = [0, 0, 0]  # No movement needed if wood is centered

        # Execute action
        obs, reward, done, info = env.step(action)

        # Render the environment so you can see the image
        env.render(mode='rgb_array')  # Efficient rendering to avoid lag
        print("display")
        time.sleep(200)  # Delay to allow you to see the environment
        
        # Break the loop to stop the agent after finding wood
        print("Wood found! Agent will stop here.")
        break  # Stop the agent after finding a tree

    else:
        print("No wood detected. Continuing to search.")
        # Continue to search until wood is found
        obs, reward, done, info = env.step([0, 0, 0])  # Stay in place if no wood detected
        time.sleep(0.05)  # Speed up the search
