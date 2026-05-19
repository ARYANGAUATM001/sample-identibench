import torch
import torch.nn as nn

from mamba_ssm import Mamba


class Model(nn.Module):

    def __init__(
            self,
            input_dim=1,
            d_model=16,
            num_classes=1
    ):

        super().__init__()

        # Project dataset input features
        # into Mamba hidden dimension
        self.input_proj = nn.Linear(
            input_dim,
            d_model
        )

        # Single Mamba block
        # closest to original baseline idea
        self.mamba = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        # Output prediction head
        self.head = nn.Linear(
            d_model,
            num_classes
        )

    def forward(self, x):

        # (B, L, input_dim)
        x = self.input_proj(x)

        # (B, L, d_model)
        x = self.mamba(x)

        # (B, L, 1)
        x = self.head(x)

        # (B, L)
        return x.squeeze(-1)
