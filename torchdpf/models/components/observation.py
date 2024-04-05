import numpy as np 
import torch 
from torch import nn
from torchdpf.models.initializations import init_xavier_normal


class ObservationalEncoder(nn.Module): 
    """
    Observational encoder for the DPF model. Described on page 3 of the paper, labeled h_theta. 
    """
    def __init__(self, H, W, in_channels=3, embedding_dim=128, dropout_rate=0.3): 
        super(ObservationalEncoder, self).__init__()
        
        """
            Args: 
                H: Height of the input image
                W: Width of the input image
                in_channels: Number of channels in the input image
                dropout_rate: Dropout rate for the dropout layer in the encoder
        
            Observational encoder has the following model structure 
            conv(3x3, 16, stride 2, relu)
            conv(3x3, 32, stride 2, relu)
            conv(3x3, 64, stride 2, relu)
            dropout(keep 0.3)
            fc(128, relu)
        """
        
        conv1_num_filters = 16
        conv2_num_filters = 32
        conv3_num_filters = 64
        fc_num_features = 128
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=conv1_num_filters, 
                               kernel_size=3, 
                               stride=2, 
                               padding='same')
        self.conv2 = nn.Conv2d(in_channels=conv1_num_filters, 
                               out_channels=conv2_num_filters, 
                               kernel_size=3, 
                               stride=2, 
                               padding='same')
        self.conv3 = nn.Conv2d(in_channels=conv2_num_filters,
                               out_channels=conv3_num_filters,
                               kernel_size=3,
                               stride=2,
                               padding='same')
        self.dropout = nn.Dropout(p=dropout_rate)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(in_features= conv3_num_filters * (H//8) * (W//8), 
                            out_features=fc_num_features)
        self.relu = nn.ReLU()
        
        # Initialize weights using Xavier normal initialization
        self.apply(self.init_xavier_normal)
        
    def forward(self, x):
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.relu(self.fc(x))
        
        return x
    



class ObservationalLikelihoodEstimator(nn.Module): 
    """
    Observational likelihood estimator for the DPF model. Described on page 3 of the paper, labeled l_theta.
    """
    def __init__(self, in_features=128, min_observation_likelihood=0.004): 
        super(ObservationalLikelihoodEstimator, self).__init__()
        """
            Observational likelihood estimator has the following model structure 
            fc(128, relu)
            fc(128, relu)
            fc(1, sigmoid)
            BUG: in_features is probably going to be 128 + 3 since observation + particle state
        """
        
        fc1_num_features = 128
        fc2_num_features = 128
        fc3_num_features = 1
        
        self.min_observation_likelihood = min_observation_likelihood
        self.fc1 = nn.Linear(in_features=in_features, 
                             out_features=fc1_num_features)
        self.fc2 = nn.Linear(in_features=fc1_num_features, 
                             out_features=fc2_num_features)
        self.fc3 = nn.Linear(in_features=fc2_num_features,
                                out_features=fc3_num_features)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights using Xavier normal initialization
        self.apply(self.init_xavier_normal)
        
    def forward(self, x): 
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

        # Scale the output to be in the range [min_observation_likelihood, 1]
        # This is to preserve numerical stability
        x = x * (1 - self.min_observation_likelihood) + self.min_observation_likelihood
        return x