import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import os

# Load processed data
def load_processed_data(filename="processed_data.pkl"):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print("Loaded processed data.")
    print("Processed data keys:", data.keys())  # Show available keys
    
    return data

# simple fully connected neural network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x)) 
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  
        return x

# Custom dataset class to load demonstrations
class DemoDataset(Dataset):
    def __init__(self, observations, actions):
        self.observations = observations
        self.actions = actions

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = torch.tensor(self.observations[idx], dtype=torch.float32)
        action = torch.tensor(self.actions[idx], dtype=torch.long)
        return obs, action

# Training loop for the behavioral cloning model
def train_bc_model(model, observations, actions_data, num_epochs=50, batch_size=64):
    # Create dataset
    demo_dataset = DemoDataset(observations, actions_data)
    dataloader = DataLoader(demo_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        model.train() 
        total_loss = 0
        
        for batch in dataloader:
            obs, action = batch
            optimizer.zero_grad() 

            # Forward pass: predict actions from observations
            action_pred = model(obs)
            loss = loss_fn(action_pred, action)
            total_loss += loss.item()

            # Backward pass: update weights
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

# Save the model after training
def save_model(model, filename="bc_model.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


def main():
    print("Loading processed data ...")
    data = load_processed_data()

    # Extract observations and actions from the processed data
    observations = np.array(data['observations'])
    actions = np.array(data['actions'])

    # Debug
    print(f"Observations shape: {observations.shape}")
    print(f"Actions shape: {actions.shape}")
    
    # Flatten observations and actions to match input dimensions
    observations = observations.reshape(-1, observations.shape[-1])  # Flattening observations
    actions = actions.reshape(-1) 
    
    # Check the new shapes
    print(f"Flattened Observations shape: {observations.shape}")
    print(f"Flattened Actions shape: {actions.shape}")

    # Define the input and output dimensions for the neural network
    input_dim = observations.shape[1]  # Number of features 
    output_dim = len(actions) 
    
    # Initialize the model
    model = PolicyNetwork(input_dim, output_dim)

    # Train the model
    print("Training BC model ...")
    train_bc_model(model, observations, actions, num_epochs=50, batch_size=64)

    # Save the trained model
    save_model(model)

if __name__ == "__main__":
    main()
