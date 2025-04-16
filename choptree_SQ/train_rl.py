import gym
import minerl
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy

from obs_wrapper import FlattenObservationWrapper
from action_wrapper import DictToMultiDiscreteWrapper
from reward_wrapper import LogRewardWrapper

# Define the environment wrapper
def make_env():
    env = gym.make("MineRLObtainDiamondShovel-v0")

    # Apply wrappers
    env = DictToMultiDiscreteWrapper(env)
    env = FlattenObservationWrapper(env)  # This should flatten observations correctly
    env = LogRewardWrapper(env)  # Add reward wrapper here

    return env

# Define the BC policy network architecture (matches the trained BC model)
class BCPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BCPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # First layer
        self.fc2 = nn.Linear(256, 128)       # Second layer
        self.fc3 = nn.Linear(128, output_dim)  # Output layer for action prediction

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation to fc1 output
        x = F.relu(self.fc2(x))  # Apply ReLU activation to fc2 output
        return self.fc3(x)  # Output action logits

# Define the custom PPO policy with both the actor (BC model) and critic (value network)
class CustomPPOPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Input and output dimensions
        input_dim = kwargs["observation_space"].shape[0]  # Flattened observation space size (12289)
        output_dim = kwargs["action_space"].n  # Action space size (48325)
        
        # Define the actor (BC model)
        self.actor = BCPolicy(input_dim=input_dim, output_dim=output_dim)
        
        # Define the critic (value network), simple MLP for state value prediction
        self.critic_fc1 = nn.Linear(input_dim, 256)
        self.critic_fc2 = nn.Linear(256, 128)
        self.critic_fc3 = nn.Linear(128, 1)  # Output a single value (state value)
        
    def forward(self, obs):
        # Actor (BC model)
        actions = self.actor(obs)
        
        # Critic (value network)
        value = F.relu(self.critic_fc1(obs))
        value = F.relu(self.critic_fc2(value))
        value = self.critic_fc3(value)  # Single value representing state value
        
        return actions, value

def train():
    # Wrap the environment with DummyVecEnv
    env = DummyVecEnv([make_env])

    # Define the input and output dimensions based on your pre-trained BC model
    input_dim = 12289  # Size of flattened observation space
    output_dim = 48325  # Size of action space (this will depend on your environment)

    # Create the BC model (pretrained model)
    bc_policy = BCPolicy(input_dim=input_dim, output_dim=output_dim)

    # Load the BC model weights (assuming the model is saved as 'bc_model.pth')
    bc_model = torch.load("bc_model.pth")
    
    # Load the state dictionary into the BC policy
    bc_policy.load_state_dict(bc_model)

    # Set up the PPO model with the custom policy as the starting point
    model = PPO(CustomPPOPolicy, env, verbose=1, tensorboard_log="./ppo_choptree_tensorboard")

    # Initialize PPO model with BC policy as the starting point
    model.policy.actor.load_state_dict(bc_policy.state_dict())  # Transfer BC model weights to PPO's actor

    print("[INFO] Starting training...")
    model.learn(total_timesteps=10000)  # You can adjust this as needed
    model.save("ppo_choptree_model")

    env.close()

if __name__ == "__main__":
    train()
