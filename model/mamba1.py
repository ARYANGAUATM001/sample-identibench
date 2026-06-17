import torch
import torch.nn as nn

from mamba_ssm import Mamba


class Model(nn.Module):

    def __init__(
        self,
        input_dim=1,
        d_model=64,
        num_classes=1
    ):

        super().__init__()

        self.input_proj = nn.Linear(
            input_dim,
            d_model
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.mamba = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.head = nn.Linear(
            d_model,
            num_classes
        )

    def forward(self, x):

        # (B, L, input_dim)
        x = self.input_proj(x)

        # residual branch
        residual = x

        # pre-norm
        x = self.norm(x)

        # mamba block
        x = self.mamba(x)

        # residual connection
        x = x + residual

        # prediction head
        x = self.head(x)

        return x.squeeze(-1)
