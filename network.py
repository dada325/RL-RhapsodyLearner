### this is not in the framwork
#
#old file for inference.
#
###


import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F

# Base class for defining common functions across all networks
class NetworkBase:
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Defines the directory for storing checkpoints
        self.checkpoint_dir = kwargs['chkpt_dir']
        # Defines the checkpoint file path
        self.checkpoint_file = os.path.join(self.checkpoint_dir, kwargs['name'])
        # Name of the network
        self.name = kwargs['name']
        
    # Function for saving the current state of the network as a checkpoint
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
    # Function for loading a previously saved checkpoint
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        

# Class for a simple linear network architecture
class LinearBase(NetworkBase, nn.Module):
    def __init__(self, name, chkpt_dir, input_dims, hidden_dims=[256]):
        super().__init__(name = name, chkpt_dir = chkpt_dir)
        # First fully connected layer
        self.fc1 = nn.Linear(input_dims, hidden_dims[0])
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[0])
        # Determining the device for computations
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    # Function for forward pass through the network
    def forward(self, state):
        f1 = F.relu(self.fc1(state))
        f2 = F.relu(self.fc2(f1))
        return f2
    

# Class for network architecture suitable for Atari games
class AtariBase(NetworkBase, nn.Module):
    def __init__(self, name, checkpoint_dir, input_dims, channels = [32, 64, 64],
                 kernels =[8, 4, 3], strides = [4, 2, 1]):
        super().__init__(name = name, chkpt_dir = chkpt_dir)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_dims[0], channels[0], kernels[0], strides[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernels[1], strides[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernels[2], strides[2])
        
        # Flattening layer
        self.flat = nn.Flatten()
        
    # Function for forward pass through the network
    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = self.flat(conv3)
        
        return conv_state
    

# Class for network architecture with a single hidden layer and output for Q-values
class Qhead(NetworkBase, nn.Module):
    def __init__(self, name, chkpt_dir, input_dims,
                 n_actions, hidden_layer=[512]):
        super().__init__(name=name, chkpt_dir = chkpt_dir)
        
        # First fully connected layer
        self.fc1 = nn.Linear(input_dims, hidden_layer[0])
        # Second fully connected layer with output size equal to number of possible actions
        self.fc2 = nn.Linear(hidden_layer[0], n_actions)
        
        # Determining the device for computations
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    # Function for forward pass through the network
    def forward(self, state):
        f1 = F.relu(self.fc1(state))
        # Calculate Q-values
        q_values = self.fc2(f1)
        
        return q_values
