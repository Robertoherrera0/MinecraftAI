import pickle
from action_wrapper import DictToMultiDiscreteWrapper
import gym
import minerl

number = "003"
with open("human-demonstrations/episode_"+number+"/meta.pkl", "rb") as f:
    meta = pickle.load(f)

with open("human-demonstrations/episode_"+number+"/actions.pkl", "rb") as f:
    data = pickle.load(f)

start = 10
end = 100

print("\n")
for item, value in meta.items():
    print(f" {item}: {value}")

print("\nStep Data:")
for i, step in enumerate(data[start:end]):
    print(f"Step {i + start}:")
    print("  Action:", step["action"])
    print("  Total Reward:", step["total_reward"])
    print()