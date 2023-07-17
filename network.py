import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F

class NetworkBase:
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.checkpoint_dir = kwargs['chkpt_dir']
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            kwargs['name'])
        self.name = kwargs['name']
        
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        
        
class LinearBase(NetworkBase, nn.Module):
    def __init__(self, name, chkpt_dir, input_dims, hidden_dims[256]):
        super().__init__(name = name, chkpt_dir = checkpoint_dir)
        self.fc1 = nn.Linear(*input_dims, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[0])
        self.device = T.device('cuda:0' if T.cuda.is_available(), else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        f1 = F.relu(self.fc1(state))
        f2 = F.relu(self.fc2(f1))
        return f2
    
    
class AtariBase(NetworkBase, nn.Module):
    def __init__(self, name, checkpoint_dir, input_dims, channels = (32, 64, 64),
                 kernels =(8, 4, 3), strides = (4, 2, 1)):
        super().__init__(name = name, chkpt_dir = chkpt_dir)
        
        assert len(channels) == 3
        assert len(kernels) == 3
        assert len(strides) == 3
        
        self.conv1 = nn.Conv2d(input_dims[0], channels[0], kernels[0],
                               strides[0])
        self.conv2 = nn.Conv2d()
        self.conv3 = nn.Conv2d()
        self.flat = nn.Flatten()
        
    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = self.flat(conv3)
        
        return conv_state
    

class Qhead(NetworkBase, nn.Module):
    def __init__(self, name, chkpt_dir, input_dims,
                 n_actions, hidden_layer=[512]):
        super().__init__(name=name, chkpt_dir = chkpt_dir)
        assert len(hidden_layer) == 1
        
        self.fc1 = nn.Linear(*input_dims, hidden_layer[0])
        self.fc2 = nn.Linear(hidden_layer[0], n_actions)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        f1 = F.relu(self.fc1(state))
        q_values = self.fc2(f1)
        
        return q_values