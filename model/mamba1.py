import torch
import torch.nn as nn

from mamba_ssm import Mamba


class MambaBlock(nn.Module):

    def __init__(
        self,
        d_model,
        d_state,
        d_conv=4,
        expand=2,
    ):
        super().__init__()

        self.norm = nn.RMSNorm(d_model)

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

        return x + residual


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

        self.input_proj = nn.Linear(
            input_dim,
            d_model
        )

        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.RMSNorm(
            d_model
        )

        self.head = nn.Linear(
            d_model,
            num_classes
        )

    def forward(self, x):

        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        x = self.head(x)

        if x.shape[-1] == 1:
            x = x.squeeze(-1)

        return x
