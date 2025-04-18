import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

def load_processed_data(filename="processed_good.pkl"):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print("Loaded processed data.")
    return data

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DemoDataset(Dataset):
    def __init__(self, observations, actions):
        self.observations = observations
        self.actions = actions

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = torch.tensor(self.observations[idx], dtype=torch.float32)
        action = torch.tensor(self.actions[idx], dtype=torch.float32)
        return obs, action

def train_bc_model(model, observations, actions_data, num_epochs=50, batch_size=64):
    demo_dataset = DemoDataset(observations, actions_data)
    dataloader = DataLoader(demo_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0
        for obs, action in dataloader:
            optimizer.zero_grad()
            action_pred = model(obs)
            loss = loss_fn(action_pred, action)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

def save_model(model, filename="bc_model.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def main():
    data = load_processed_data()
    observations = np.array(data['observations'])
    actions = np.array(data['actions'])

    # Shape enforcement
    observations = observations.reshape(len(observations), -1)
    actions = actions.reshape(len(actions), -1)
    print("Obs shape:", observations.shape)
    print("Action shape:", actions.shape)

    input_dim = observations.shape[1]
    output_dim = actions.shape[1]
    model = PolicyNetwork(input_dim, output_dim)

    train_bc_model(model, observations, actions, num_epochs=50, batch_size=64)
    save_model(model)


if __name__ == "__main__":
    main()
