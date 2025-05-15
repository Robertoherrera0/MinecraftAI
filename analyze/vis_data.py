import numpy as np

filename = "data/processed_data/processed_good_cnn.npz"
start = 50000
end = 50001

data = np.load(filename)

print("\nLoaded keys from file:")
for key in data.files:
    print(f" - {key}: shape = {data[key].shape}, dtype = {data[key].dtype}")

print("\nSample steps:")
for i in range(start, min(end, len(data['actions']))):
    print(f"\nStep {i}:")
    print("  Action:", data['actions'][i])
    print("  Reward:", data['rewards'][i])
    print("  Done:", data['done_flags'][i])
    print("  Total Reward:", data['total_rewards'][i])
    print("  Inventory:", data['inv'][i])
    print("  POV shape:", data['pov'][i].shape)
