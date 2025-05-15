import pickle
import os

# Path to your episode file
episode_path = "../data/human-demonstrations/episode_001/actions.pkl"

# Load the episode
with open(episode_path, "rb") as f:
    episode = pickle.load(f)

print(f"Loaded {len(episode)} transitions")

prev_log_count = 0

for i, transition in enumerate(episode):
    inv = transition["obs"]["inv"]
    log_count = inv[0] if isinstance(inv, list) or inv.shape[0] == 1 else inv.get("log", 0)

    if log_count > prev_log_count:
        print(f"[STEP {i}] Log picked up!")
        print(f"Inventory before: {prev_log_count}, after: {log_count}")
        print(f"Full Inventory: {inv}")
        print("-" * 40)

    prev_log_count = log_count
