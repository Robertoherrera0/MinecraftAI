import pickle
import numpy as np
import cv2
import os

def load_data():
    actions = []
    for episode_number in range(99, 100): # adjust 
        episode_path = f"human-demonstrations/episode_{episode_number:03d}"
        with open(f"{episode_path}/actions.pkl", "rb") as f:
            episode_actions = pickle.load(f)
        actions.append(episode_actions)
    return actions

# Preprocess the data for each episode
def preprocess_data(actions):
    all_obs_data = []
    all_action_data = []
    all_reward_data = []
    all_done_data = []
    all_total_reward_data = []
    all_is_good_episode_data = []

    # Process each episode
    for episode in actions:
        episode_obs_data = []
        episode_action_data = []
        episode_reward_data = []
        episode_done_data = []
        episode_total_reward_data = []
        episode_is_good_episode_data = []

        # Process each transition in the episode
        for transition in episode:
            pov = transition['obs']['pov'] 
            pov = pov[..., :3]  # Take only RGB channels if needed
            pov = cv2.resize(pov, (64, 64)).flatten()  # Resize and flatten the image

            inventory = transition['obs']['inv']  
            action = transition['action']  
            reward = transition['reward']
            done = transition['done']
            total_reward = transition['total_reward']

            # Flatten the inventory to be consistent
            inventory = np.array(inventory).flatten()

            # Concatenate pov and inventory to form the observation input
            obs = np.concatenate([pov, inventory])

            # Append data for each transition
            episode_obs_data.append(obs)
            episode_action_data.append(action)
            episode_reward_data.append(reward)
            episode_done_data.append(done)
            episode_total_reward_data.append(total_reward)
            episode_is_good_episode_data.append(transition.get('is_good_episode', 0))  # Assuming we get this flag

        # Append the processed data for the episode
        all_obs_data.append(episode_obs_data)
        all_action_data.append(episode_action_data)
        all_reward_data.append(episode_reward_data)
        all_done_data.append(episode_done_data)
        all_total_reward_data.append(episode_total_reward_data)
        all_is_good_episode_data.append(episode_is_good_episode_data)

    # Convert lists to NumPy arrays
    observations = np.array(all_obs_data, dtype=np.float32)
    actions = np.array(all_action_data, dtype=np.int64)
    rewards = np.array(all_reward_data, dtype=np.float32)
    done_flags = np.array(all_done_data, dtype=np.bool_)
    total_rewards = np.array(all_total_reward_data, dtype=np.float32)
    is_good_episode = np.array(all_is_good_episode_data, dtype=np.bool_)

    return {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'done_flags': done_flags,
        'total_rewards': total_rewards,
        'is_good_episode': is_good_episode
    }


def save_processed_data(data, filename="processed_data.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")

def main():
    print("Loading data ...")
    actions = load_data()

    print("Processing data ...")
    processed_data = preprocess_data(actions)

    print("Saving processed data ...")
    save_processed_data(processed_data)

if __name__ == "__main__":
    main()
