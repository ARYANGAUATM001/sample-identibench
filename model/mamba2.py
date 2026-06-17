import torch
import torch.nn as nn
from mamba_ssm import Mamba2


class Model(nn.Module):

     def __init__(
        self,
        input_dim=1,
        d_model=128,
        d_state=64,
        n_layers=2,
        num_classes=1
    ):

        self.input_proj = nn.Linear(2, 128)

        self.norm1 = nn.LayerNorm(128)
        self.norm2 = nn.LayerNorm(128)

        self.mamba1 = Mamba2(
            d_model=128,
            d_state=64
        )

        self.mamba2 = Mamba2(
            d_model=128,
            d_state=64
        )

        self.output = nn.Linear(128, 1)

    def forward(self, u, y_prev):

        y_prev = y_prev.unsqueeze(-1)

        x = torch.cat([u, y_prev], dim=-1)

        x = self.input_proj(x)

        residual = x
        x = self.norm1(x)
        x = self.mamba1(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.mamba2(x)
        x = x + residual

        x = self.output(x)

        return x.squeeze(-1)
