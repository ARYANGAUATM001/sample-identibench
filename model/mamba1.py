import torch
import torch.nn as nn
from mamba_ssm import Mamba

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(2, 128)
        self.mamba = Mamba(d_model=128)
        self.output_layer = nn.Linear(128, 1)

    def forward(self, u, y_prev):

        y_prev = y_prev.unsqueeze(-1)

        x = torch.cat([u, y_prev], dim=-1)

        x = self.input_layer(x)

        x = self.mamba(x)

        return self.output_layer(x).squeeze(-1)
