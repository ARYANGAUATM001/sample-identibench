import torch

def train_model(model, train_data, valid_data, config, device):
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = torch.nn.MSELoss()

    SEQ_LEN = 1000

    for ep in range(config["epochs"]):
        model.train()
        tr = 0

        for (u, y, _) in train_data:
            u = torch.tensor(u, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

            if y.ndim == 3:
                y = y.squeeze(-1)

            for i in range(0, len(u), SEQ_LEN):
                u_batch = u[i:i+SEQ_LEN].unsqueeze(0).to(device)
                y_batch = y[i:i+SEQ_LEN].unsqueeze(0).to(device)

                pred = model(u_batch)
                loss = loss_fn(pred, y_batch)

                opt.zero_grad()
                loss.backward()
                opt.step()

                tr += loss.item()

        model.eval()
        vl = 0

        with torch.no_grad():
            for (u, y, _) in valid_data:
                u = torch.tensor(u, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)

                if y.ndim == 3:
                    y = y.squeeze(-1)

                u = u[:SEQ_LEN].unsqueeze(0).to(device)
                y = y[:SEQ_LEN].unsqueeze(0).to(device)

                vl += loss_fn(model(u), y).item()

        print(f"{ep} | {tr:.3f} | {vl:.3f}")

    return model