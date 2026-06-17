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

        # --------------------------------------------------
        # Input = [u_t , y_{t-1}]
        # --------------------------------------------------

        self.input_proj = nn.Linear(
            input_dim + 1,
            d_model
        )

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

        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(
            d_model
        )

        self.head = nn.Linear(
            d_model,
            num_classes
        )

    def forward(
        self,
        u,
        y_prev
    ):
        """
        u       : (B, L, input_dim)
        y_prev  : (B, L)
        """

        if y_prev.ndim == 2:
            y_prev = y_prev.unsqueeze(-1)

        # --------------------------------------------------
        # Concatenate previous output history
        # --------------------------------------------------

        x = torch.cat(
            [u, y_prev],
            dim=-1
        )

        x = self.input_proj(x)

        # --------------------------------------------------
        # Residual Mamba stack
        # --------------------------------------------------

        for norm, layer in zip(
            self.norms,
            self.layers
        ):

            residual = x

            x = norm(x)

            x = layer(x)

            x = x + residual

        x = self.final_norm(x)

        x = self.head(x)

        return x.squeeze(-1)
