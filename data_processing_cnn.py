import pickle
import numpy as np
import cv2
import os

# Load good and bad episodes from saved demonstration folders
def load_data():
    good_episodes = []
    bad_episodes = []
    root_dir = "data/human-demonstrations"

    for episode_dir in sorted(os.listdir(root_dir)):
        full_path = os.path.join(root_dir, episode_dir)

        if not os.path.isdir(full_path):
            continue

        meta_path = os.path.join(full_path, "meta.pkl")
        actions_path = os.path.join(full_path, "actions.pkl")

        if not os.path.exists(meta_path) or not os.path.exists(actions_path):
            continue

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        with open(actions_path, 'rb') as f:
            episode_actions = pickle.load(f)

        if meta.get("is_good_episode", False):
            good_episodes.append(episode_actions)
        else:
            bad_episodes.append(episode_actions)

    return good_episodes, bad_episodes

# Format raw episode data into numpy arrays
def preprocess_data(episodes):
    pov_list = []
    inv_list = []
    action_list = []
    reward_list = []
    done_list = []
    total_reward_list = []

    for episode in episodes:
        for transition in episode:
            # Prepare POV image (3, 64, 64)
            pov = transition['obs']['pov'][..., :3]  # Ensure RGB only
            pov = cv2.resize(pov, (64, 64)).astype(np.uint8)
            pov = np.transpose(pov, (2, 0, 1))  # Convert to (C, H, W)

            # Inventory is already flat
            inventory = np.array(transition['obs']['inv'], dtype=np.float32)

            # Get the rest
            action = transition['action']
            reward = transition['reward']
            done = transition['done']
            total_reward = transition['total_reward']

            pov_list.append(pov)
            inv_list.append(inventory)
            action_list.append(action)
            reward_list.append(reward)
            done_list.append(done)
            total_reward_list.append(total_reward)

    return {
        'pov': np.array(pov_list, dtype=np.uint8),              # (N, 3, 64, 64)
        'inv': np.array(inv_list, dtype=np.float32),            # (N, ?)
        'actions': np.array(action_list, dtype=np.int64),       # (N, 8)
        'rewards': np.array(reward_list, dtype=np.float32),
        'done_flags': np.array(done_list, dtype=np.bool_),
        'total_rewards': np.array(total_reward_list, dtype=np.float32)
    }

# Save preprocessed data to compressed .npz file
def save_processed_data(data, filename):
    np.savez(filename, **data)
    print(f" Data saved to: {filename}")

# Entry point
def main():
    print(" Loading episodes ...")
    good_eps, bad_eps = load_data()
    print(f"Good episodes: {len(good_eps)}")
    print(f"Bad episodes: {len(bad_eps)}")

    if good_eps:
        print("Processing good episodes ...")
        processed_good = preprocess_data(good_eps)
        save_processed_data(processed_good, "data/processed_data/processed_good_cnn.npz")

    if bad_eps:
        print("Processing bad episodes ...")
        processed_bad = preprocess_data(bad_eps)
        save_processed_data(processed_bad, "data/processed_data/processed_bad_cnn.npz")

if __name__ == "__main__":
    main()
