"""Linear feed-through / BLA skip helpers shared by the models.

Only depends on torch (no mamba_ssm), so mamba3 stays importable on CPU.
"""
import torch.nn as nn
import torch.nn.functional as F


def make_skip(input_dim, num_classes, bla_taps):
    """bla_taps>1 -> learnable causal FIR (a linear Best-Linear-Approximation
    of the dynamics); else an instantaneous linear feed-through. Zero-init."""
    if bla_taps and bla_taps > 1:
        m = nn.Conv1d(input_dim, num_classes, kernel_size=bla_taps, bias=True)
    else:
        m = nn.Linear(input_dim, num_classes)
    nn.init.zeros_(m.weight)
    nn.init.zeros_(m.bias)
    return m


def apply_skip(skip, bla_taps, u):
    """u: (B, L, input_dim) -> (B, L, num_classes)."""
    if bla_taps and bla_taps > 1:
        x = u.transpose(1, 2)               # (B, D, L)
        x = F.pad(x, (bla_taps - 1, 0))     # left-pad for causality
        x = skip(x)                         # (B, num_classes, L)
        return x.transpose(1, 2)            # (B, L, num_classes)
    return skip(u)
