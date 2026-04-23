class SelectiveSSM(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()

        self.hidden_dim = hidden_dim


        self.B = nn.Linear(input_dim, hidden_dim)


        self.C = nn.Linear(hidden_dim, 1)


        self.gate = nn.Linear(input_dim, hidden_dim)
        self.A_net = nn.Linear(input_dim, hidden_dim)

    def forward(self, u):
        B, T, _ = u.shape

        h = torch.zeros(B, self.hidden_dim, device=u.device)
        y_out = []

        for t in range(T):
            u_t = u[:, t]

            x = self.B(u_t)


            A = torch.tanh(self.A_net(u_t))


            g = torch.sigmoid(self.gate(u_t))


            h = g * (A * h) + (1 - g) * x

            y = self.C(h)
            y_out.append(y)

        return torch.stack(y_out, dim=1).squeeze(-1)
