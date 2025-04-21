import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor  # type: ignore
from train_bc_cnn import CNNPolicyNetwork
import gym.spaces

class BCFeatureExtractorCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=128):
        super().__init__(observation_space, features_dim)

        # Extract shape info
        pov_shape = observation_space['pov'].shape  # (64, 64, 3)
        inv_dim = observation_space['inv'].shape[0]  # usually 1
        action_dim = 8  # fixed

        # Load pre-trained encoder (from Behavioral Cloning)
        self.encoder = CNNPolicyNetwork(inv_dim=inv_dim, action_dim=action_dim)
        self.encoder.load_state_dict(torch.load("models/bc_model_cnn.pth"))
        self.encoder.eval()

        # Freeze weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Keep the CNN and the FC stack up to the last Linear layer
        self.cnn = self.encoder.cnn
        self.fc = nn.Sequential(*list(self.encoder.fc.children())[:-1])  # Drop final output

        # Set final features dim manually
        self._features_dim = features_dim

    def forward(self, observations):
        pov = observations['pov'].float() / 255.0  # Normalize image
        inv = observations['inv'].float()

        with torch.no_grad():
            cnn_out = self.cnn(pov)
            x = torch.cat([cnn_out, inv], dim=1)
            return self.fc(x)
