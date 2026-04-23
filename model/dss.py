import torch
import torch.nn as nn


class SimpleSSM(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()

        self.A = nn.Parameter(torch.ones(hidden_dim) * 0.8)
        self.B = nn.Linear(input_dim, hidden_dim)
        self.C = nn.Linear(hidden_dim, 1)

    def forward(self, u):
        B, T, _ = u.shape

        h = torch.zeros(B, self.A.shape[0], device=u.device)
        y_out = []

        for t in range(T):
            h = self.A * h + self.B(u[:, t])
            y = self.C(h)
            y_out.append(y)

        return torch.stack(y_out, dim=1).squeeze(-1)