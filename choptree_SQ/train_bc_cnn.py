import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os

def load_processed_data(filename="processed_data/processed_good_cnn.npz"):
    data = np.load(filename)
    print("Loaded processed data.")
    return data

class CNNPolicyNetwork(nn.Module):
    def __init__(self, inv_dim, action_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # (32, 15, 15)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # (64, 6, 6)
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 6 * 6 + inv_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, pov, inv):
        x = self.cnn(pov)
        x = torch.cat([x, inv], dim=1)
        return self.fc(x)

class DemoDataset(Dataset):
    def __init__(self, povs, invs, actions):
        self.povs = povs
        self.invs = invs
        self.actions = actions

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        pov = torch.tensor(self.povs[idx], dtype=torch.float32) / 255.0  # normalize
        inv = torch.tensor(self.invs[idx], dtype=torch.float32)
        action = torch.tensor(self.actions[idx], dtype=torch.float32)
        return pov, inv, action

def train_bc_model(model, povs, invs, actions_data, num_epochs=200, batch_size=64):
    dataset = DemoDataset(povs, invs, actions_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for pov, inv, action in dataloader:
            optimizer.zero_grad()
            action_pred = model(pov, inv)
            loss = loss_fn(action_pred, action)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

    # Plot log-scale loss
    plt.figure()
    plt.semilogy(epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("BC Loss Over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def save_model(model, filename="models/bc_model_cnn.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def main():
    data = load_processed_data()
    pov = np.array(data['pov'])              # shape (N, 3, 64, 64)
    inv = np.array(data['inv'])              # shape (N, 1) or more
    actions = np.array(data['actions'])      # shape (N, 8)

    print("POV shape:", pov.shape)
    print("INV shape:", inv.shape)
    print("Action shape:", actions.shape)

    inv_dim = inv.shape[1]
    action_dim = actions.shape[1]

    model = CNNPolicyNetwork(inv_dim, action_dim)
    if os.path.exists("models/bc_model_cnn.pth"):
        print("Loading existing BC model...")
        model.load_state_dict(torch.load("models/bc_model_cnn.pth"))

    train_bc_model(model, pov, inv, actions, num_epochs=50, batch_size=64)
    save_model(model)

if __name__ == "__main__":
    main()
