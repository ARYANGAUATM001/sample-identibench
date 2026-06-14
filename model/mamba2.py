import torch
import torch.nn as nn

# TRUE Mamba-2 layer
from mamba_ssm.modules.mamba2 import Mamba2


# ============================================================
# Mamba-2 Block
# ============================================================

class Mamba2Block(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=64,
        expand=2,
        d_conv=4,
        dropout=0.0,
    ):
        super().__init__()

        # ----------------------------------------------------
        # PreNorm (official style)
        # ----------------------------------------------------

        self.norm = nn.RMSNorm(d_model)

        # ----------------------------------------------------
        # True Mamba-2 SSD layer
        #
        # headdim is chosen so that nheads = expand*d_model/headdim
        # is a multiple of 8. causal_conv1d's channel-last kernel
        # otherwise raises a stride-alignment error (e.g. the default
        # headdim=64 gives nheads=4 for d_model=128 and fails).
        # ----------------------------------------------------

        d_inner = expand * d_model
        headdim = d_inner // 8  # -> nheads = 8

        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )

        # ----------------------------------------------------
        # Optional dropout
        # ----------------------------------------------------

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # residual
        residual = x

        # prenorm
        x = self.norm(x)

        # Mamba-2
        x = self.mamba(x)

        # dropout
        x = self.dropout(x)

        # residual connection
        x = x + residual

        return x


# ============================================================
# Full Model
# ============================================================

class Model(nn.Module):

    def __init__(
        self,
        input_dim=1,
        d_model=128,
        d_state=64,
        n_layers=6,
        expand=2,
        d_conv=4,
        num_classes=1,
        dropout=0.0,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Input projection
        # ----------------------------------------------------

        self.input_proj = nn.Linear(
            input_dim,
            d_model
        )

        # ----------------------------------------------------
        # Stacked Mamba-2 blocks
        # ----------------------------------------------------

        self.layers = nn.ModuleList(
            [
                Mamba2Block(
                    d_model=d_model,
                    d_state=d_state,
                    expand=expand,
                    d_conv=d_conv,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        self.final_norm = nn.RMSNorm(
            d_model
        )

        # ----------------------------------------------------
        # Output head
        # ----------------------------------------------------

        self.head = nn.Linear(
            d_model,
            num_classes
        )

        # Direct linear feed-through (BLA-style): captures the linear
        # input->output term so the SSM only learns the nonlinear residual.
        self.skip = nn.Linear(input_dim, num_classes)
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)

    def forward(self, x):

        """
        x shape:
            (B, L, input_dim)
        """

        u = x

        # input projection
        x = self.input_proj(x)

        # stacked Mamba-2 layers
        for layer in self.layers:
            x = layer(x)

        # final norm
        x = self.final_norm(x)

        # output projection = nonlinear head + linear skip
        x = self.head(x) + self.skip(u)

        # safe squeeze for regression/binary
        if x.shape[-1] == 1:
            x = x.squeeze(-1)

        return x
