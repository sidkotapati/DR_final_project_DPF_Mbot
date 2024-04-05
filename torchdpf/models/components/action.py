import numpy as np 
import torch 
from torch import nn
from torchdpf.models.initializations import init_xavier_normal  


class ActionSampler(nn.Module): 
    """
    Action sampler for the DPF model. Described on page 3 of the paper, labeled f_theta.
    
    In this implementation, a random noise vector is generated, but rather than added to the actions, 
    it is used to generate the noise. This is because the noise is generated using a neural network.
    This is why you will see the noise being generated in the forward method, concatinated with the actions,
    and then passed through the neural network to generate the noise.

    """
    def __init__(self, action_dim=3, state_dim=3, noise_dim=None): 
        super(ActionSampler, self).__init__()
        
        """
            Action sampler has the following model structure 
            2 x fc(32, relu), fc(3) + mean centering across particles
        """
        
        self.action_dim = action_dim 
        self.state_dim = state_dim 
        
        if noise_dim is None: 
            self.noise_dim = action_dim
        else: 
            self.noise_dim = noise_dim
        
        fc1_num_features = 32
        fc2_num_features = 32
        fc3_num_features = action_dim
        
        self.fc1 = nn.Linear(in_features=self.action_dim + self.noise_dim, 
                             out_features=fc1_num_features)
        
        self.fc2 = nn.Linear(in_features=fc1_num_features,
                            out_features=fc2_num_features)
        
        self.fc3 = nn.Linear(in_features=fc2_num_features,
                            out_features=self.state_dim)
        
        self.relu = nn.ReLU()
        
        # Initialize weights using Xavier normal initialization
        self.apply(self.init_xavier_normal)
        
    def generate_motion_noise(self, x): 
        """
        Generate motion noise for the action sampler. 
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
    def forward(self, actions, stds, particles): 
        # Normalize actions 
        actions_normalized = actions / stds[:, None]
        
        # Add dimension for particles and repeat actions 
        actions_expanded = (actions_normalized.unsqueeze(1)
                            .expand(-1, particles.shape[1], -1))
        
        # Generate random input 
        random_input = torch.randn_like(actions_expanded)
        
        # Concatenate actions and random input
        x = torch.cat([actions_expanded, random_input], dim=-1)
        
        # Generate action noise
        delta = self.generate_motion_noise(x)
        
        # Detach gradient from delta 
        delta = delta.detach()
        
        # Zero-mean the action noise
        delta -= delta.mean(dim=1, keepdim=True)
        
        # Add noise to actions 
        noisy_actions = actions.unsqueeze(1) + delta
        
        return noisy_actions



class MotionTransitionModel(nn.Module): 
    """
    Motion transition model for the DPF model. Described on page 3 of the paper, labeled g_theta.
    """
    def __init__(self, state_dim=3, action_dim=3, noise_dim=None): 
        super(MotionTransitionModel, self).__init__()
        
        """
            Motion transition model has the following model structure 
            2 x fc(32, relu), fc(3)
        """
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        fc1_num_features = 128
        fc2_num_features = 128
        fc3_num_features = 128
        fc4_num_features = state_dim
        
        self.fc1 = nn.Linear(in_features=self.state_dim + self.action_dim, 
                             out_features=fc1_num_features)
        
        self.fc2 = nn.Linear(in_features=fc1_num_features,
                            out_features=fc2_num_features)
        
        self.fc3 = nn.Linear(in_features=fc2_num_features,
                            out_features=fc3_num_features)
        self.fc4 = nn.Linear(in_features=fc3_num_features, out_features=fc4_num_features)
        
        self.relu = nn.ReLU()
        
        # Initialize weights using Xavier normal initialization
        self.apply(self.init_xavier_normal)
        
    def forward(self, states, actions): 
        x = torch.cat([states, actions], dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        
        return x