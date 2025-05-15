import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor #type:ignore
from bc_model.train_bc import PolicyNetwork

INPUT_DIM = 64 * 64 * 3 + 1
OUTPUT_DIM = 8

class BCFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.net = PolicyNetwork(INPUT_DIM, OUTPUT_DIM)
        self.net.load_state_dict(torch.load("choptree/models/bc_model.pth"))
        self.net.eval()

    def forward(self, x):
        return self.net.fc2(torch.relu(self.net.fc1(x)))
