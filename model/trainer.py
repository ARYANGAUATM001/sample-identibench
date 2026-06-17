import torch
import torch.nn as nn

def train_model(
    model,
    train_data,
    valid_data,
    config,
    device
):

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    loss_fn = nn.MSELoss()

    best_valid_loss = float("inf")

    for epoch in range(config["epochs"]):

        # -------------------------
        # TRAIN
        # -------------------------

        model.train()

        for u, y, _ in train_data:
            

            u = u.float().to(device)
            y = y.float().to(device)

            if y.ndim == 3:
                y = y.squeeze(-1)

            y_prev = torch.zeros_like(y)
            y_prev[:, 1:] 

            optimizer.zero_grad()

            pred = model(u, y_prev)

            loss = loss_fn(pred, y)

            loss.backward()

        # -------------------------
        # VALIDATION
        # -------------------------

        model.eval()

        valid_loss = 0

        with torch.no_grad():

            for u, y, _ in valid_data:

                u = u.float().to(device)
                y = y.float().to(device)

                if y.ndim == 3:
                    y = y.squeeze(-1)

                y_prev = torch.zeros_like(y

                pred = model(u, y_prev)

                loss = loss_fn(pred, y)

                valid_loss += loss.item()

        valid_loss /= len(valid_data)

        print(
            f"Epoch {epoch+1} "
            f"Valid Loss: {valid_loss:.6f}"
        )

        if valid_loss < best_valid_loss:

            best_valid_loss = valid_loss

            torch.save(
                model.state_dict(),
                "best_model.pt"
            )

    return model
