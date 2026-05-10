import torch
import torch.nn as nn
from mamba_ssm import Mamba


def rotate_half(x):

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    return torch.cat(
        (-x2, x1),
        dim=-1
    )


class RotaryEmbedding(nn.Module):

    def __init__(self, dim):

        super().__init__()

        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(0, dim, 2).float()
                / dim
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq
        )

    def forward(self, x):

        B, T, D = x.shape

        t = torch.arange(
            T,
            device=x.device
        ).type_as(self.inv_freq)

        freqs = torch.einsum(
            "i,j->ij",
            t,
            self.inv_freq
        )

        emb = torch.cat(
            [freqs, freqs],
            dim=-1
        )

        cos = emb.cos()[None, :, :]
        sin = emb.sin()[None, :, :]

        return (
            (x * cos)
            + (rotate_half(x) * sin)
        )


class SwiGLU(nn.Module):

    def __init__(
        self,
        dim,
        hidden_dim
    ):
        super().__init__()

        self.w1 = nn.Linear(
            dim,
            hidden_dim
        )

        self.w2 = nn.Linear(
            dim,
            hidden_dim
        )

        self.w3 = nn.Linear(
            hidden_dim,
            dim
        )

    def forward(self, x):

        return self.w3(
            torch.nn.functional.silu(
                self.w1(x)
            )
            * self.w2(x)
        )


class Mamba3Block(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=128,
        expand=4,
    ):
        super().__init__()

        # RMSNorm instead of LayerNorm
        self.norm1 = nn.RMSNorm(
            d_model
        )

        # Mamba block
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=1,      # Mamba-3 minimizes conv
            expand=expand,
        )

        # Second norm
        self.norm2 = nn.RMSNorm(
            d_model
        )

        # SwiGLU FFN
        self.ffn = SwiGLU(
            dim=d_model,
            hidden_dim=d_model * 4
        )

    def forward(self, x):

        # Mamba residual
        residual = x

        x = self.norm1(x)

        x = self.mamba(x)

        x = x + residual

        # FFN residual
        residual = x

        x = self.norm2(x)

        x = self.ffn(x)

        x = x + residual

        return x


class Model(nn.Module):

    def __init__(
        self,
        input_dim,
        d_model=128,
        d_state=128,
        n_layers=8,
        num_classes=1,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(
            input_dim,
            d_model
        )

        # Rotary embeddings
        self.rope = RotaryEmbedding(
            d_model
        )

        # Deep stacked architecture
        self.layers = nn.ModuleList(
            [
                Mamba3Block(
                    d_model=d_model,
                    d_state=d_state,
                )
                for _ in range(n_layers)
            ]
        )

        # Final RMSNorm
        self.final_norm = nn.RMSNorm(
            d_model
        )

        # Output head
        self.head = nn.Linear(
            d_model,
            num_classes
        )

    def forward(self, x):

        # (B, L, input_dim)
        x = self.input_proj(x)

        # rotary positional encoding
        x = self.rope(x)

        # stacked Mamba-3 blocks
        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        x = self.head(x)

        # safe squeeze
        if x.shape[-1] == 1:
            x = x.squeeze(-1)

        return x
