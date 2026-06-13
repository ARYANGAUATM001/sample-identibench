import torch
import torch.nn as nn

from mamba_ssm import Mamba


# ============================================================
# Mamba-1 Block (PreNorm + residual)
# ============================================================

class Mamba1Block(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.0,
    ):
        super().__init__()

        self.norm = nn.RMSNorm(d_model)

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.norm(x)

        x = self.mamba(x)

        x = self.dropout(x)

        return x + residual


# ============================================================
# Full Model (stacked Mamba-1)
# ============================================================

class Model(nn.Module):

    def __init__(
        self,
        input_dim=1,
        d_model=128,
        d_state=16,
        n_layers=6,
        expand=2,
        d_conv=4,
        num_classes=1,
        dropout=0.0,
    ):

        super().__init__()

        # Project dataset input features
        # into Mamba hidden dimension
        self.input_proj = nn.Linear(
            input_dim,
            d_model
        )

        # Stacked Mamba-1 blocks
        self.layers = nn.ModuleList(
            [
                Mamba1Block(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.RMSNorm(d_model)

        # Output prediction head
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

        x = self.final_norm(x)

        # (B, L, num_classes)
        x = self.head(x)

        # (B, L) for regression
        if x.shape[-1] == 1:
            x = x.squeeze(-1)

        return x
