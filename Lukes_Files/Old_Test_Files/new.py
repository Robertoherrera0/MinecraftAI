import minerl
import gym

def test_minerl():
    try:
        env = gym.make("MineRLNavigateDense-v0")
        obs = env.reset()

        print("MineRL environment loaded successfully!")

        # Take a few random actions to verify functionality
        for _ in range(5):
            action = env.action_space.sample()  # Sample a random action
            obs, reward, done, _ = env.step(action)
            print(f"Step Reward: {reward}, Done: {done}")
            if done:
                env.reset()

        env.close()
        print("MineRL test completed successfully!")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_minerl()


# data = minerl.data.make(
#     'MineRLObtainDiamond-v0')


# for current_state, action, reward, next_state, done \
#     in data.batch_iter(
#         batch_size=1, num_epochs=1, seq_len=32):

#         # Print the POV @ the first step of the sequence
#         print(current_state['pov'][0])

#         # Print the final reward pf the sequence!
#         print(reward[-1])

#         # Check if final (next_state) is terminal.
#         print(done[-1])

#         # ... do something with the data.
#         print("At the end of trajectories the length"
#               "can be < max_sequence_len", len(reward))

# import os
# import minerl
# import gym

# # Check if MINERL_DATA_ROOT is set correctly
# data_root = os.getenv("MINERL_DATA_ROOT")
# if data_root is None:
#     print("Error: MINERL_DATA_ROOT is not set. Please set it to your dataset directory.")
#     exit(1)

# print(f"Using dataset path: {data_root}")

# # Try to load the dataset
# try:
#     data = minerl.data.make('MineRLObtainDiamond-v0')
#     print("MineRL dataset loaded successfully!")
# except Exception as e:
#     print(f"Error loading dataset: {e}")
#     exit(1)

# # Iterate through a small batch of the dataset
# print("\nReading dataset...\n")
# try:
#     for current_state, action, reward, next_state, done in data.batch_iter(batch_size=1, num_epochs=1, seq_len=32):
#         # Print the first frame of the sequence
#         print("First frame POV:", current_state['pov'][0].shape)

#         # Print the last reward in the sequence
#         print("Final reward:", reward[-1])

#         # Print if the last state in the sequence is terminal
#         print("Is terminal:", done[-1])

#         # Print sequence length (should be ≤ seq_len)
#         print("Sequence length:", len(reward))
        
#         # Stop after the first batch (optional)
#         break
# except Exception as e:
#     print(f"Error while iterating over dataset: {e}")
