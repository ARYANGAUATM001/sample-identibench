import torch
import torch.nn as nn

from mamba_ssm import Mamba


class Model(nn.Module):

    def __init__(
        self,
        input_dim=1,
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

        # Stacked Mamba layers
        self.layers = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=4,
                    expand=2,
                )
                for _ in range(n_layers)
            ]
        )

        # Output head
        self.head = nn.Linear(
            d_model,
            num_classes
        )

    def forward(self, x):

        # (B, L, input_dim)
        x = self.input_proj(x)

        # (B, L, d_model)
        for layer in self.layers:
            x = layer(x)

        # (B, L, 1)
        x = self.head(x)

        # (B, L)
        if x.shape[-1] == 1:
            x = x.squeeze(-1)

        return x
