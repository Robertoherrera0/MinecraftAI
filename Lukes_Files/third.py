# # import gym
# # import minerl
# # import numpy as np

# # # Define the RGB values for each wood type
# # WOOD_COLORS = {
# #     "wood_1": np.array([166, 157, 111]),  # #a69d6f
# #     "wood_2": np.array([145, 117, 77]),   # #91754d
# #     "wood_3": np.array([99, 73, 43]),     # #63492b
# #     "wood_4": np.array([54, 35, 16]),     # #362310
# #     "wood_5": np.array([142, 104, 79])    # #8e684f
# # }

# # # Function to check if a pixel's color is within a tolerance range
# # def is_similar_color(pixel, target_color, tolerance=30):
# #     return np.all(np.abs(pixel - target_color) <= tolerance)

# # # Initialize the environment
# # env = gym.make("MineRLObtainDiamondShovel-v0")  # Modify to your environment
# # obs = env.reset()

# # # Look through the field of view (pov) to find matching wood blocks
# # wood_found = {wood: 0 for wood in WOOD_COLORS}  # Dictionary to track found woods

# # # Iterate through the observation's pov (image)
# # pov_image = obs['pov']  # pov is typically a 3D array (height, width, channels)

# # for row in pov_image:
# #     for pixel in row:
# #         for wood, target_color in WOOD_COLORS.items():
# #             if is_similar_color(pixel, target_color):
# #                 wood_found[wood] += 1

# # # Print out the results
# # for wood, count in wood_found.items():
# #     print(f"{wood}: {count} pixels found.")

# # # Close the environment
# # env.close()

# import gym
# import minerl
# import numpy as np
# import time

# # Define the RGB values for each wood type
# WOOD_COLORS = {
#     "wood_1": np.array([166, 157, 111]),  # #a69d6f
#     "wood_2": np.array([145, 117, 77]),   # #91754d
#     "wood_3": np.array([99, 73, 43]),     # #63492b
#     "wood_4": np.array([54, 35, 16]),     # #362310
#     "wood_5": np.array([142, 104, 79])    # #8e684f
# }

# # Function to check if a pixel's color is within a tolerance range
# def is_similar_color(pixel, target_color, tolerance=30):
#     return np.all(np.abs(pixel - target_color) <= tolerance)

# # Initialize the environment
# env = gym.make("MineRLObtainDiamondShovel-v0")  # Modify to your environment
# obs = env.reset()

# # Function to move the agent in the given direction (forward, backward, left, right)
# def move_agent(action):
#     action_dict = {
#         'forward': [1, 0, 0],  # Move forward
#         'backward': [-1, 0, 0], # Move backward
#         'left': [0, 1, 0],      # Move left
#         'right': [0, -1, 0],    # Move right
#     }
#     return action_dict.get(action, [0, 0, 0])

# # Look through the field of view (pov) to find matching wood blocks
# wood_found = {wood: 0 for wood in WOOD_COLORS}  # Dictionary to track found woods
# wood_positions = []

# # Iterate through the observation's pov (image)
# pov_image = obs['pov']  # pov is typically a 3D array (height, width, channels)

# # Collect the positions of the detected wood
# height, width, _ = pov_image.shape
# for i in range(height):
#     for j in range(width):
#         pixel = pov_image[i, j]
#         for wood, target_color in WOOD_COLORS.items():
#             if is_similar_color(pixel, target_color):
#                 wood_positions.append((i, j))
#                 wood_found[wood] += 1

# # Print out the results
# for wood, count in wood_found.items():
#     print(f"{wood}: {count} pixels found.")

# # If we detected wood, move towards it
# if wood_positions:
#     # Here, we'll just take the first detected wood position for simplicity
#     target_position = wood_positions[0]  # (i, j)

#     # Calculate a basic movement strategy (move towards the target position)
#     target_x, target_y = target_position
#     print(f"Detected wood at position: {target_position}")

#     # Check if the target is in the center of the image (simple approach)
#     if target_x < height // 2:
#         print("Move forward")
#         action = move_agent('forward')
#     elif target_x > height // 2:
#         print("Move backward")
#         action = move_agent('backward')
#     else:
#         action = [0, 0, 0]  # No movement needed if wood is centered

#     # Execute action
#     obs, reward, done, info = env.step(action)

#     time.sleep(0.1)  # Small delay between steps to simulate agent action
# else:
#     print("No wood detected in the current field of view.")

# # Close the environment
# env.close()

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

# Look through the field of view (pov) to find matching wood blocks
wood_found = {wood: 0 for wood in WOOD_COLORS}  # Dictionary to track found woods
wood_positions = []

# Iterate through the observation's pov (image)
pov_image = obs['pov']  # pov is typically a 3D array (height, width, channels)

# Collect the positions of the detected wood
height, width, _ = pov_image.shape
for i in range(height):
    for j in range(width):
        pixel = pov_image[i, j]
        for wood, target_color in WOOD_COLORS.items():
            if is_similar_color(pixel, target_color):
                wood_positions.append((i, j))
                wood_found[wood] += 1

# Print out the results
for wood, count in wood_found.items():
    print(f"{wood}: {count} pixels found.")

# If we detected wood, move towards it
if wood_positions:
    # Here, we'll just take the first detected wood position for simplicity
    target_position = wood_positions[0]  # (i, j)

    # Calculate a basic movement strategy (move towards the target position)
    target_x, target_y = target_position
    print(f"Detected wood at position: {target_position}")

    # Check if the target is in the center of the image (simple approach)
    if target_x < height // 2:
        print("Move forward")
        action = move_agent('forward')
    elif target_x > height // 2:
        print("Move backward")
        action = move_agent('backward')
    else:
        action = [0, 0, 0]  # No movement needed if wood is centered

    # Execute action
    obs, reward, done, info = env.step(action)

    time.sleep(0.1)  # Small delay between steps to simulate agent action
else:
    print("No wood detected in the current field of view.")

# Render the environment after taking the step
env.render()
i = 0
while (i < 10000):
    i = i + 1
# Close the environment
env.close()
