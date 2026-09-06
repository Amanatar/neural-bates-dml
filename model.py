import torch
import torch.nn as nn

class BatesNeuralSurrogate(nn.Module):
    def __init__(self, input_dim=10, output_dim=55):
        super(BatesNeuralSurrogate, self).__init__()
        
        # 4 hidden layers, ELU activation (smooth derivatives for autograd)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)
