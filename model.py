import torch
import torch.nn as nn

class BatesNeuralSurrogate(nn.Module):
    """
    Neural Surrogate for Bates SVJD Option Pricing and Differential ML (Sobolev training).
    
    Employs smooth, twice-differentiable activations (SiLU / GELU / Softplus) to ensure
    differentiable first-order sensitivities (Delta, Vega, Theta, Rho) and non-vanishing
    Gamma and vol-of-vol surface curvatures.
    """
    def __init__(self, input_dim=10, output_dim=1, activation='silu'):
        super(BatesNeuralSurrogate, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = str(activation).lower()
        
        if self.activation_name == 'silu':
            act_fn = nn.SiLU
        elif self.activation_name == 'gelu':
            act_fn = nn.GELU
        elif self.activation_name == 'softplus':
            act_fn = nn.Softplus
        elif self.activation_name == 'elu':
            act_fn = nn.ELU
        else:
            act_fn = nn.SiLU
            
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            act_fn(),
            nn.Linear(128, 256),
            act_fn(),
            nn.Linear(256, 256),
            act_fn(),
            nn.Linear(256, 128),
            act_fn(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)
