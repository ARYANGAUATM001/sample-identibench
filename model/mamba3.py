import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utilities
# ============================================================

def rotate_half(x):

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    out = torch.stack(
        (-x2, x1),
        dim=-1
    )

    return out.flatten(-2)


# ============================================================
# Data-dependent Rotary Embedding
# (Mamba-3 complex SSM approximation)
# ============================================================

class DataDependentRoPE(nn.Module):

    def __init__(self, d_model):

        super().__init__()

        assert d_model % 2 == 0

        self.theta_proj = nn.Linear(
            d_model,
            d_model // 2
        )

    def forward(self, x):

        """
        x: (B, L, D)
        """

        theta = self.theta_proj(x)

        cos = torch.cos(theta)
        sin = torch.sin(theta)

        cos = torch.repeat_interleave(
            cos,
            2,
            dim=-1
        )

        sin = torch.repeat_interleave(
            sin,
            2,
            dim=-1
        )

        return (
            x * cos
            + rotate_half(x) * sin
        )


# ============================================================
# BC / QK Normalization
# ============================================================

class BCNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):

        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(dim)
        )

    def forward(self, x):

        norm = torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True)
            + self.eps
        )

        return x * norm * self.weight


# ============================================================
# SwiGLU
# ============================================================

class SwiGLU(nn.Module):

    def __init__(self, dim, hidden_dim):

        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):

        return self.w3(
            F.silu(self.w1(x))
            * self.w2(x)
        )


# ============================================================
# Mamba-3 SSM Layer
# Implements:
# - exponential-trapezoidal recurrence
# - data-dependent rotations
# - BCNorm
# - learnable B/C biases
# ============================================================

class Mamba3SSM(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=128,
    ):

        super().__init__()

        self.d_model = d_model
        self.d_state = d_state

        # ----------------------------------
        # projections
        # ----------------------------------

        self.dt_proj = nn.Linear(
            d_model,
            1
        )

        self.A_proj = nn.Linear(
            d_model,
            1
        )

        self.lambda_proj = nn.Linear(
            d_model,
            1
        )

        self.B_proj = nn.Linear(
            d_model,
            d_state
        )

        self.C_proj = nn.Linear(
            d_model,
            d_state
        )

        # ----------------------------------
        # BC normalization
        # ----------------------------------

        self.b_norm = BCNorm(d_state)
        self.c_norm = BCNorm(d_state)

        # ----------------------------------
        # learnable biases
        # ----------------------------------

        self.B_bias = nn.Parameter(
            torch.zeros(d_state)
        )

        self.C_bias = nn.Parameter(
            torch.zeros(d_state)
        )

        # ----------------------------------
        # complex-state approximation
        # using data-dependent rotary
        # ----------------------------------

        self.rope_B = DataDependentRoPE(
            d_state
        )

        self.rope_C = DataDependentRoPE(
            d_state
        )

        # ----------------------------------
        # output projection
        # ----------------------------------

        self.out_proj = nn.Linear(
            d_state,
            d_model
        )

    def forward(self, x):

        """
        x: (B, L, D)
        """

        B, L, D = x.shape

        # ----------------------------------
        # state
        # ----------------------------------

        h = torch.zeros(
            B,
            self.d_state,
            device=x.device,
            dtype=x.dtype
        )

        outputs = []

        prev_Bx = torch.zeros(
            B,
            self.d_state,
            device=x.device,
            dtype=x.dtype
        )

        for t in range(L):

            xt = x[:, t]

            # ----------------------------------
            # dynamic parameters
            # ----------------------------------

            dt = F.softplus(
                self.dt_proj(xt)
            )

            A = -F.softplus(
                self.A_proj(xt)
            )

            lam = torch.sigmoid(
                self.lambda_proj(xt)
            )

            # ----------------------------------
            # exponential-trapezoidal coeffs
            # ----------------------------------

            alpha = torch.exp(dt * A)

            beta = (
                (1.0 - lam)
                * dt
                * alpha
            )

            gamma = lam * dt

            # ----------------------------------
            # B / C projections
            # ----------------------------------

            B_t = self.B_proj(xt)
            C_t = self.C_proj(xt)

            # ----------------------------------
            # BCNorm
            # ----------------------------------

            B_t = self.b_norm(B_t)
            C_t = self.c_norm(C_t)

            # ----------------------------------
            # biases
            # ----------------------------------

            B_t = B_t + self.B_bias
            C_t = C_t + self.C_bias

            # ----------------------------------
            # complex SSM approximation
            # with data-dependent rotations
            # ----------------------------------

            B_t = self.rope_B(
                B_t.unsqueeze(1)
            ).squeeze(1)

            C_t = self.rope_C(
                C_t.unsqueeze(1)
            ).squeeze(1)

            # ----------------------------------
            # state input
            # ----------------------------------

            Bx = B_t * xt.mean(
                dim=-1,
                keepdim=True
            )

            # ----------------------------------
            # Mamba-3 recurrence
            #
            # h_t =
            #   alpha * h_{t-1}
            # + beta  * B_{t-1}x_{t-1}
            # + gamma * B_t x_t
            # ----------------------------------

            h = (
                alpha * h
                + beta * prev_Bx
                + gamma * Bx
            )

            # output
            y = h * C_t

            y = self.out_proj(y)

            outputs.append(y)

            prev_Bx = Bx

        return torch.stack(
            outputs,
            dim=1
        )


# ============================================================
# Full Mamba-3 Block
# ============================================================

class Mamba3Block(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=128,
        expand=4,
    ):

        super().__init__()

        self.norm1 = nn.RMSNorm(
            d_model
        )

        self.ssm = Mamba3SSM(
            d_model=d_model,
            d_state=d_state,
        )

        self.norm2 = nn.RMSNorm(
            d_model
        )

        self.ffn = SwiGLU(
            d_model,
            d_model * expand
        )

    def forward(self, x):

        residual = x

        x = self.norm1(x)

        x = self.ssm(x)

        x = x + residual

        residual = x

        x = self.norm2(x)

        x = self.ffn(x)

        x = x + residual

        return x


# ============================================================
# Full Model
# ============================================================

class Model(nn.Module):

    def __init__(
        self,
        input_dim=1,
        d_model=256,
        d_state=128,
        n_layers=8,
        num_classes=1,
    ):

        super().__init__()

        self.input_proj = nn.Linear(
            input_dim,
            d_model
        )

        self.layers = nn.ModuleList(
            [
                Mamba3Block(
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
