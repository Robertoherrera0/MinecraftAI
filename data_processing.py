import pickle
import numpy as np
import cv2
import os

def load_data():
    good_episodes = []
    bad_episodes = []
    root_dir = "human-demonstrations"

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

def preprocess_data(episodes):
    all_obs_data = []
    all_action_data = []
    all_reward_data = []
    all_done_data = []
    all_total_reward_data = []
    all_is_good_episode_data = []

    for episode in episodes:
        for transition in episode:
            pov = transition['obs']['pov'][..., :3]
            pov = cv2.resize(pov, (64, 64)).flatten()

            inventory = np.array(transition['obs']['inv']).flatten()
            obs = np.concatenate([pov, inventory])

            action = transition['action']
            reward = transition['reward']
            done = transition['done']
            total_reward = transition['total_reward']
            is_good = transition.get('is_good_episode', 0)

            all_obs_data.append(obs)
            all_action_data.append(action)
            all_reward_data.append(reward)
            all_done_data.append(done)
            all_total_reward_data.append(total_reward)
            all_is_good_episode_data.append(is_good)

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

def save_processed_data(data, filename):
    np.savez(filename, **data)
    print(f" Data saved to: {filename}")

def main():
    print(" Loading episodes ...")
    good_eps, bad_eps = load_data()
    print(f"Good episodes: {len(good_eps)}")
    print(f"Bad episodes: {len(bad_eps)}")

    if good_eps:
        print("Processing good episodes ...")
        processed_good = preprocess_data(good_eps)
        save_processed_data(processed_good, "processed_data/processed_good.npz")

    if bad_eps:
        print("Processing bad episodes ...")
        processed_bad = preprocess_data(bad_eps)
        save_processed_data(processed_bad, "processed_data/processed_bad.npz")

if __name__ == "__main__":
    main()
