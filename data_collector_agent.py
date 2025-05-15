import minerl  # type: ignore
import gym
import os
import numpy as np
import pickle
import re
import cv2
from pynput import keyboard  # type: ignore
from choptree.wrappers.custom_reward_wrapper import CustomRewardWrapper

# Config
INVENTORY_KEYS = ["log"]
CAMERA_BINS = 21
CAMERA_CENTER = CAMERA_BINS // 2
CREDIT_WINDOW = 10
CAMERA_RANGE = 10.0
ACTION_KEYS = ["attack", "back", "forward", "jump", "left", "right"]

# Create environment
base_env = gym.make("MineRLObtainDiamondShovel-v0")
env = CustomRewardWrapper(base_env)
obs = env.reset()

# Create directory for saving new episode
root_dir = "data/human-demonstrations"
os.makedirs(root_dir, exist_ok=True)
existing = [d for d in os.listdir(root_dir) if re.match(r'episode_\d{3}', d)]
numbers = [int(re.search(r'\d{3}', d).group()) for d in existing]
next_number = max(numbers) + 1 if numbers else 1
episode_name = f"episode_{next_number:03d}"
episode_dir = os.path.join(root_dir, episode_name)
os.makedirs(episode_dir)

# Key tracking and camera speed
keys_pressed = set()
exit_flag = False
CAMERA_SPEED = 2.0

# Convert key inputs to MineRL-style action dict
def get_action():
    act = {k: 0 for k in ACTION_KEYS}
    if 'w' in keys_pressed: act['forward'] = 1
    if 'a' in keys_pressed: act['left'] = 1
    if 'd' in keys_pressed: act['right'] = 1
    if 's' in keys_pressed: act['back'] = 1
    if 'b' in keys_pressed: act['jump'] = 1
    if 'm' in keys_pressed: act['attack'] = 1

    yaw = 0.0
    pitch = 0.0
    if 'j' in keys_pressed: yaw -= CAMERA_SPEED
    if 'l' in keys_pressed: yaw += CAMERA_SPEED
    if 'i' in keys_pressed: pitch -= CAMERA_SPEED
    if 'k' in keys_pressed: pitch += CAMERA_SPEED
    act['camera'] = np.array([pitch, yaw], dtype=np.float32)
    return act

# Process raw observation into POV + inventory
def flatten_observation(obs):
    pov = obs["pov"][..., :3]
    pov = cv2.resize(pov, (64, 64)).astype(np.uint8)
    inventory = obs.get("inventory", {})
    inv_vec = np.array([inventory.get(k, 0) for k in INVENTORY_KEYS], dtype=np.float32)
    return {"pov": pov, "inv": inv_vec}

# Convert action dict to MultiDiscrete array format
def dict_to_multidiscrete(action):
    action_vec = [action[k] for k in ACTION_KEYS]

    pitch = action["camera"][0]
    yaw = action["camera"][1]

    pitch_bin = int(round((pitch / CAMERA_RANGE) * (CAMERA_CENTER))) + CAMERA_CENTER
    yaw_bin   = int(round((yaw   / CAMERA_RANGE) * (CAMERA_CENTER))) + CAMERA_CENTER

    pitch_bin = int(np.clip(np.round(pitch) + CAMERA_CENTER, 0, CAMERA_BINS - 1))
    yaw_bin = int(np.clip(np.round(yaw) + CAMERA_CENTER, 0, CAMERA_BINS - 1))
    
    action_vec.extend([yaw_bin, pitch_bin])
    return np.array(action_vec, dtype=np.int32)

# Keyboard event handlers
def on_press(key):
    global exit_flag
    try:
        if key.char == 'q':
            exit_flag = False
        else:
            keys_pressed.add(key.char)
    except AttributeError:
        pass

def on_release(key):
    try:
        keys_pressed.discard(key.char)
    except AttributeError:
        pass

# Start keyboard listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

print("\nControlling the agent. Press 'q' to quit and save the episode.\n"
      "w/a/s/d: move | b: jump | m: attack\n"
      "i/j/k/l: camera | q: quit and save\n")

# Main loop for human control and demonstration recording
actions = []
total_reward = 0

while not exit_flag and len(actions) < 1500:
    raw_action = get_action()
    raw_next_obs, reward, done, _ = env.step(raw_action)

    # Credit future reward to past steps
    if reward > 0:
        for i in range(max(0, len(actions) - CREDIT_WINDOW), len(actions)):
            actions[i]["reward"] += reward / CREDIT_WINDOW

    total_reward += reward

    wrapped_obs = flatten_observation(obs)
    wrapped_next_obs = flatten_observation(raw_next_obs)
    wrapped_action = dict_to_multidiscrete(raw_action)

    actions.append({
        "obs": wrapped_obs,
        "action": wrapped_action,
        "reward": reward,
        "next_obs": wrapped_next_obs,
        "done": done,
        "total_reward": total_reward
    })

    obs = raw_next_obs
    env.render()

    if len(actions) >= 500:
        print("Reached 500 steps, exiting...")
        break

    if done:
        obs = env.reset()

env.close()

# Save actions
with open(os.path.join(episode_dir, "actions.pkl"), "wb") as f:
    pickle.dump(actions, f)
print(f"Episode saved to: {episode_dir}/actions.pkl")

# Ask user if this was a good episode
label = input("Was this a good episode? (y/n): ").strip().lower()
is_good = (label == "y")

# Save metadata
meta = {
    "total_reward": total_reward,
    "is_good_episode": is_good,
    "num_steps": len(actions)
}

with open(os.path.join(episode_dir, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)
print(f"Metadata saved to: {episode_dir}/meta.pkl")
print(f"Total number of steps taken: {len(actions)}")
