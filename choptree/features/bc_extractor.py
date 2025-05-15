import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor  # type:ignore
from bc_model.train_bc import PolicyNetwork

# Flattened image size (64x64 RGB) + 1 inventory value
INPUT_DIM = 64 * 64 * 3 + 1
OUTPUT_DIM = 8

# Feature extractor using layers from a pre-trained BC MLP model
class BCFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        # Load the pre-trained BC model
        self.net = PolicyNetwork(INPUT_DIM, OUTPUT_DIM)
        self.net.load_state_dict(torch.load("choptree/models/bc_model.pth"))
        self.net.eval()

    def forward(self, x):
        # Pass through first two layers of the BC model (skip final output layer)
        return self.net.fc2(torch.relu(self.net.fc1(x)))
