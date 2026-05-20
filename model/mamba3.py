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
# ============================================================

class DataDependentRoPE(nn.Module):

    def __init__(self, dim):

        super().__init__()

        assert dim % 2 == 0

        self.theta_proj = nn.Linear(
            dim,
            dim // 2
        )

    def forward(self, x):

        """
        x: (..., D)
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
# BCNorm / QKNorm
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
# Mamba-3 SSM
# ============================================================

class Mamba3SSM(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=128,
    ):

        super().__init__()

        assert d_state % 2 == 0

        self.d_model = d_model
        self.d_state = d_state

        # ----------------------------------------------------
        # Dynamic parameter projections
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # SSM projections
        # ----------------------------------------------------

        self.B_proj = nn.Linear(
            d_model,
            d_state
        )

        self.C_proj = nn.Linear(
            d_model,
            d_state
        )

        # IMPORTANT FIX:
        # learned input projection
        self.x_proj = nn.Linear(
            d_model,
            d_state
        )

        # ----------------------------------------------------
        # BCNorm
        # ----------------------------------------------------

        self.b_norm = BCNorm(d_state)
        self.c_norm = BCNorm(d_state)

        # ----------------------------------------------------
        # Learnable biases
        # ----------------------------------------------------

        self.B_bias = nn.Parameter(
            torch.zeros(d_state)
        )

        self.C_bias = nn.Parameter(
            torch.zeros(d_state)
        )

        # ----------------------------------------------------
        # Data-dependent rotary embeddings
        # ----------------------------------------------------

        self.rope_B = DataDependentRoPE(
            d_state
        )

        self.rope_C = DataDependentRoPE(
            d_state
        )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        self.out_proj = nn.Linear(
            d_state,
            d_model
        )

    def forward(self, x):

        """
        x:
            (B, L, D)
        """

        B, L, D = x.shape

        # ----------------------------------------------------
        # Initial hidden state
        # ----------------------------------------------------

        h = torch.zeros(
            B,
            self.d_state,
            device=x.device,
            dtype=x.dtype
        )

        prev_Bx = torch.zeros(
            B,
            self.d_state,
            device=x.device,
            dtype=x.dtype
        )

        outputs = []

        # ----------------------------------------------------
        # Sequential recurrence
        # ----------------------------------------------------

        for t in range(L):

            xt = x[:, t]

            # ------------------------------------------------
            # Dynamic SSM parameters
            # ------------------------------------------------

            dt = F.softplus(
                self.dt_proj(xt)
            )

            A = -F.softplus(
                self.A_proj(xt)
            )

            lam = torch.sigmoid(
                self.lambda_proj(xt)
            )

            # ------------------------------------------------
            # Exponential-trapezoidal coefficients
            # ------------------------------------------------

            alpha = torch.exp(
                torch.clamp(
                    dt * A,
                    min=-20.0,
                    max=5.0
                )
            )

            beta = (
                (1.0 - lam)
                * dt
                * alpha
            )

            gamma = lam * dt

            # ------------------------------------------------
            # Projections
            # ------------------------------------------------

            B_t = self.B_proj(xt)
            C_t = self.C_proj(xt)

            # learned input projection
            x_proj = self.x_proj(xt)

            # ------------------------------------------------
            # BCNorm
            # ------------------------------------------------

            B_t = self.b_norm(B_t)
            C_t = self.c_norm(C_t)

            # ------------------------------------------------
            # Learnable biases
            # ------------------------------------------------

            B_t = B_t + self.B_bias
            C_t = C_t + self.C_bias

            # ------------------------------------------------
            # Data-dependent RoPE
            # ------------------------------------------------

            B_t = self.rope_B(B_t)
            C_t = self.rope_C(C_t)

            # ------------------------------------------------
            # State input
            # ------------------------------------------------

            Bx = B_t * x_proj

            # ------------------------------------------------
            # Mamba-3 recurrence
            #
            # h_t =
            #   alpha * h_{t-1}
            # + beta  * B_{t-1}x_{t-1}
            # + gamma * B_t x_t
            # ------------------------------------------------

            h = (
                alpha * h
                + beta * prev_Bx
                + gamma * Bx
            )

            # ------------------------------------------------
            # Output
            # ------------------------------------------------

            y = h * C_t

            y = self.out_proj(y)

            outputs.append(y)

            prev_Bx = Bx

        return torch.stack(
            outputs,
            dim=1
        )


# ============================================================
# Mamba-3 Block
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

        # ----------------------------------------------------
        # SSM block
        # ----------------------------------------------------

        residual = x

        x = self.norm1(x)

        x = self.ssm(x)

        x = x + residual

        # ----------------------------------------------------
        # FFN block
        # ----------------------------------------------------

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
        expand=4,
        num_classes=1,
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
        # Mamba-3 layers
        # ----------------------------------------------------

        self.layers = nn.ModuleList(
            [
                Mamba3Block(
                    d_model=d_model,
                    d_state=d_state,
                    expand=expand,
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

    def forward(self, x):

        """
        x:
            (B, L, input_dim)
        """

        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        x = self.head(x)

        if x.shape[-1] == 1:
            x = x.squeeze(-1)

        return x
