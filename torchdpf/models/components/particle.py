import numpy as np 
import torch 
from torch import nn
from torchdpf.models.initializations import init_xavier_normal


class ParticleProposer(nn.Module):
    """
    Particle proposer for the DPF model. Described on page 3 of the paper, labeled k_theta.
    """
    def __init__(self, state_dim=3, embedding_dim=128, dropout_rate=0.3): 
        super(ParticleProposer, self).__init__()
        
        """
            Particle proposer has the following model structure 
            fc(128, relu), fc(128, relu), fc(3)
        """
        
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        
        fc1_num_features = 128
        fc2_num_features = 128
        fc3_num_features = 128
        fc4_num_features = 4       

        self.fc1 = nn.Linear(in_features=self.embedding_dim, 
                            out_features=fc1_num_features)
        
        self.fc2 = nn.Linear(in_features=fc1_num_features,
                            out_features=fc2_num_features)
        
        self.fc3 = nn.Linear(in_features=fc2_num_features,
                            out_features=fc3_num_features)

        self.fc4 = nn.Linear(in_features=fc3_num_features,
                            out_features=fc4_num_features)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Initialize weights using Xavier normal initialization
        self.apply(self.init_xavier_normal)
        
    def forward(self, x): 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x) 
        x = self.tanh(x)
        
        return x