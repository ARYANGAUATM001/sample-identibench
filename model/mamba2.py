import torch
import torch.nn as nn
from mamba_ssm import Mamba


class Mamba2Block(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=64,
        expand=2,
        d_conv=4,
    ):
        super().__init__()

        # Mamba-2 style prenorm
        self.norm = nn.RMSNorm(d_model)

        # Mamba layer
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):

        residual = x

        x = self.norm(x)

        x = self.mamba(x)

        # residual connection
        x = x + residual

        return x


class Model(nn.Module):

    def __init__(
        self,
        input_dim,
        d_model=128,
        d_state=64,
        n_layers=6,
        num_classes=1,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(
            input_dim,
            d_model
        )

        # Stacked Mamba blocks
        self.layers = nn.ModuleList(
            [
                Mamba2Block(
                    d_model=d_model,
                    d_state=d_state,
                )
                for _ in range(n_layers)
            ]
        )

        # Final normalization
        self.final_norm = nn.RMSNorm(
            d_model
        )

        # Output head
        self.head = nn.Linear(
            d_model,
            num_classes
        )

    def forward(self, x):

        # x: (B, L, input_dim)

        x = self.input_proj(x)

        # stacked Mamba layers
        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        x = self.head(x)

        # only squeeze if binary/regression
        if x.shape[-1] == 1:
            x = x.squeeze(-1)

        return x
