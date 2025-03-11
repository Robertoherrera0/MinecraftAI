# import numpy as np

# data = np.load("../MineRLObtainDiamond-v0/v3_absolute_grape_changeling-6_37339-46767/rendered.npz")


# # Inspect the available keys
# print(data.files)  # e.g., ['pov', 'actions', 'rewards', 'next_pov', 'done']

# import numpy as np

# class MineRLDataset:
#     def __init__(self, file_path):
#         self.data = np.load(file_path)
#         self.keys = list(self.data.keys())
#         print("Loaded dataset with keys:", self.keys)

#     def batch_iter(self, batch_size=1, num_epochs=1, seq_len=32):
#         """
#         Iterates through data in batches.
#         """
#         total_steps = len(self.data['reward'])  # Use reward length as reference
#         for epoch in range(num_epochs):
#             for i in range(0, total_steps, seq_len):
#                 end_idx = min(i + seq_len, total_steps)
                
#                 batch = {key: self.data[key][i:end_idx] for key in self.keys}
#                 yield batch  # Returns a dictionary of batch data

# # Usage:
# dataset = MineRLDataset("../MineRLObtainDiamond-v0/v3_absolute_grape_changeling-6_37339-46767/rendered.npz")

# for batch in dataset.batch_iter(batch_size=1, num_epochs=1, seq_len=32):
#     print("Batch POV Shape:", batch.get('pov', None))  # Print POV if available
#     print("Batch Reward:", batch['reward'][-1])  # Print last reward in batch
#     break  # Just one iteration for testing


# import numpy as np

# # Path to your dataset
# DATASET_PATH = "../MineRLObtainDiamond-v0/v3_absolute_grape_changeling-6_37339-46767/rendered.npz"

# # Load dataset
# try:
#     data = np.load(DATASET_PATH)
#     print("Dataset loaded successfully!")
# except Exception as e:
#     print(f"Error loading dataset: {e}")
#     exit()

# # Display dataset keys
# print("Keys in dataset:", data.files)

# # Extract log inventory and camera actions
# if 'observation$inventory$log' in data and 'action$camera' in data:
#     log_inventory = data['observation$inventory$log']
#     camera_actions = data['action$camera']  # Camera movement (pitch/yaw)
    
#     print(f"Total frames: {len(log_inventory)}")

#     # Check for tree interaction
#     for i in range(1, len(log_inventory)):  
#         if log_inventory[i] > log_inventory[i - 1]:  # Log count increased
#             print(f"Tree likely chopped at frame {i}!")

#             # Check if the camera moved up before chopping (looking at tree)
#             if camera_actions[i - 1][0] < 0:
#                 print(f"  -> Camera was looking up before chopping.")

# else:
#     print("Log inventory or camera actions not found in dataset!")

import minerl
import gym
import numpy as np

# Load MineRL environment
env = gym.make("MineRLTreechop-v0")  # Environment for chopping trees
obs, _ = env.reset()

# Path to dataset (Update this with your actual path)
DATASET_PATH = "../MineRLObtainDiamond-v0/v3_absolute_grape_changeling-6_37339-46767/rendered.npz"

# Load dataset
try:
    data = np.load(DATASET_PATH)
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Extract relevant actions from dataset
if 'action$forward' in data and 'action$attack' in data:
    forward_actions = data['action$forward']
    attack_actions = data['action$attack']
    camera_actions = data.get('action$camera', None)  # Optional camera movement

    print("Replaying actions...")

    for i in range(len(forward_actions)):  # Replay dataset actions
        action = env.action_space.noop()  # Start with a no-op action

        action['forward'] = forward_actions[i]
        action['attack'] = attack_actions[i]

        if camera_actions is not None:  # Apply camera movement if available
            action['camera'] = camera_actions[i]

        obs, reward, done, truncated, _ = env.step(action)
        env.render()  # Renders the environment

        if done:
            print("Episode finished, resetting environment.")
            obs, _ = env.reset()

# Close environment after execution
env.close()
