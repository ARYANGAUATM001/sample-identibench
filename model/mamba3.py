import torch
import torch.nn as nn

from mamba_ssm import Mamba


class RotaryEmbedding(nn.Module):

    def __init__(self, dim):

        super().__init__()

        inv_freq = 1.0 / (
            10000 ** (
                torch.arange(0, dim, 2).float() / dim
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
            (freqs, freqs),
            dim=-1
        )

        cos = emb.cos()[None, :, :]
        sin = emb.sin()[None, :, :]

        return (x * cos) + (rotate_half(x) * sin)


def rotate_half(x):

    x1 = x[..., : x.shape[-1] // 2]

    x2 = x[..., x.shape[-1] // 2:]

    return torch.cat(
        (-x2, x1),
        dim=-1
    )


class Mamba3Block(nn.Module):

    def __init__(
            self,
            d_model,
            d_state
    ):

        super().__init__()

        # Pre-normalization
        self.norm = nn.LayerNorm(
            d_model
        )

        # Stronger Mamba block
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=1,
            expand=4,
        )

        # Feed-forward refinement
        self.ffn = nn.Sequential(

            nn.Linear(
                d_model,
                d_model * 2
            ),

            nn.GELU(),

            nn.Linear(
                d_model * 2,
                d_model
            )
        )

    def forward(self, x):

        residual = x

        x = self.norm(x)

        x = self.mamba(x)

        # Residual connection
        x = x + residual

        residual = x

        x = self.ffn(x)

        # Second residual connection
        x = x + residual

        return x


class Model(nn.Module):

    def __init__(
            self,
            input_dim,
            d_model=64,
            num_classes=1
    ):

        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(
            input_dim,
            d_model
        )

        # RoPE-inspired positional encoding
        self.rope = RotaryEmbedding(
            d_model
        )

        # Deeper architecture
        self.layers = nn.Sequential(

            Mamba3Block(
                d_model=d_model,
                d_state=64
            ),

            Mamba3Block(
                d_model=d_model,
                d_state=64
            ),

            Mamba3Block(
                d_model=d_model,
                d_state=128
            ),

            Mamba3Block(
                d_model=d_model,
                d_state=128
            ),
        )

        # Final normalization
        self.final_norm = nn.LayerNorm(
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

        # Positional rotation
        x = self.rope(x)

        # Mamba-3 inspired blocks
        x = self.layers(x)

        x = self.final_norm(x)

        # (B, L, 1)
        x = self.head(x)

        # (B, L)
        return x.squeeze(-1)