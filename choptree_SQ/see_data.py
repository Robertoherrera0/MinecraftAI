import pickle
from action_wrapper import DictToMultiDiscreteWrapper
import gym
import minerl

number = "002"
with open("human-demonstrations/episode_"+number+"/meta.pkl", "rb") as f:
    meta = pickle.load(f)

with open("human-demonstrations/episode_"+number+"/actions.pkl", "rb") as f:
    data = pickle.load(f)

start = 999
end = 1000

print("\n")
for item, value in meta.items():
    print(f" {item}: {value}")

dummy_env = DictToMultiDiscreteWrapper(gym.make("MineRLObtainDiamondShovel-v0"))
action_keys = dummy_env.action_keys + ["camera_yaw", "camera_pitch"]

print("\nAction fields:")
for idx, key in enumerate(action_keys):
    print(f" {idx}: {key}")

print("\nStep Data:")
for i, step in enumerate(data[start:end]):
    print(f"Step {i + start}:")
    print("  Action:", step["action"])
    print("  Total Reward:", step["total_reward"])
    print()