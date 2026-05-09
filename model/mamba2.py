import torch
import torch.nn as nn

from mamba_ssm import Mamba


class Mamba2Block(nn.Module):

    def __init__(
            self,
            d_model,
            d_state
    ):

        super().__init__()

        # Pre-normalization improves stability
        self.norm = nn.LayerNorm(
            d_model
        )

        # Mamba block
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=2,
        )

    def forward(self, x):

        residual = x

        x = self.norm(x)

        x = self.mamba(x)

        # Residual connection
        x = x + residual

        return x


class Model(nn.Module):

    def __init__(
            self,
            input_dim,
            d_model=32,
            num_classes=1
    ):

        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(
            input_dim,
            d_model
        )

        # Deeper stacked architecture
        # inspired by Mamba-2 scaling
        self.layers = nn.Sequential(

            Mamba2Block(
                d_model=d_model,
                d_state=16
            ),

            Mamba2Block(
                d_model=d_model,
                d_state=32
            ),

            Mamba2Block(
                d_model=d_model,
                d_state=64
            ),
        )

        # Final normalization
        self.final_norm = nn.LayerNorm(
            d_model
        )

        # Prediction head
        self.head = nn.Linear(
            d_model,
            num_classes
        )

    def forward(self, x):

        # (B, L, input_dim)
        x = self.input_proj(x)

        # (B, L, d_model)
        x = self.layers(x)

        x = self.final_norm(x)

        # (B, L, 1)
        x = self.head(x)

        # (B, L)
        return x.squeeze(-1)