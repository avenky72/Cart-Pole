"""
NN that takes in 4 numbers for the state and output 2 values of the q values for left and right 
Doesn't use an activation function for the last layer because none exactly fit the use case
"""

import torch.nn as nn

class Model(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        
    def forward(self, x):
        return self.network(x)
        