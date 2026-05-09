import os
import torch


def train_model(
        model,
        train_data,
        valid_data,
        config,
        device,
        model_name="mamba1"
):

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"]
    )

    loss_fn = torch.nn.MSELoss()

    seq_len = config.get("seq_len", 100)

    save_dir = f"outputs/{model_name}"

    os.makedirs(save_dir, exist_ok=True)

    best_valid_loss = float("inf")

    for epoch in range(config["epochs"]):

        # ================= TRAIN =================

        model.train()

        train_loss = 0.0

        for (u, y, _) in train_data:

            u = torch.tensor(
                u,
                dtype=torch.float32
            )

            y = torch.tensor(
                y,
                dtype=torch.float32
            )

            # FIX TARGET SHAPE
            if y.ndim == 3:
                y = y.squeeze(-1)

            if y.ndim == 2:
                y = y.squeeze(1)

            for i in range(0, len(u), seq_len):

                u_batch = (
                    u[i:i + seq_len]
                    .unsqueeze(0)
                    .to(device, non_blocking=True)
                )

                y_batch = (
                    y[i:i + seq_len]
                    .unsqueeze(0)
                    .to(device, non_blocking=True)
                )

                optimizer.zero_grad()

                pred = model(u_batch)

                # FIX OUTPUT SHAPE
                pred = pred.squeeze(-1)

                loss = loss_fn(
                    pred,
                    y_batch
                )

                if torch.isnan(loss):
                    print("NaN loss detected")
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

                train_loss += loss.item()

        # ================= VALID =================

        model.eval()

        valid_loss = 0.0

        with torch.no_grad():

            for (u, y, _) in valid_data:

                u = torch.tensor(
                    u,
                    dtype=torch.float32
                )

                y = torch.tensor(
                    y,
                    dtype=torch.float32
                )

                # FIX TARGET SHAPE
                if y.ndim == 3:
                    y = y.squeeze(-1)

                if y.ndim == 2:
                    y = y.squeeze(1)

                u = (
                    u[:seq_len]
                    .unsqueeze(0)
                    .to(device, non_blocking=True)
                )

                y = (
                    y[:seq_len]
                    .unsqueeze(0)
                    .to(device, non_blocking=True)
                )

                pred = model(u)

                # FIX OUTPUT SHAPE
                pred = pred.squeeze(-1)

                loss = loss_fn(
                    pred,
                    y
                )

                valid_loss += loss.item()

        # ================= AVERAGE LOSSES =================

        train_loss /= len(train_data)

        valid_loss /= len(valid_data)

        # ================= SAVE BEST =================

        if valid_loss < best_valid_loss:

            best_valid_loss = valid_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "valid_loss": valid_loss,
                    "epoch": epoch,
                },
                f"{save_dir}/best_model.pt"
            )

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Valid Loss: {valid_loss:.4f}"
        )

    return model