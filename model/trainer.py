import os
import torch
import torch.nn as nn

from torch.amp import autocast, GradScaler


def train_model(
    model,
    train_data,
    valid_data,
    config,
    device,
    model_name="mamba1"
):

    model = model.to(device)

    if config.get("compile", False):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get(
            "weight_decay",
            1e-2
        )
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=5
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(
                    config["epochs"] - 5,
                    1
                )
            )
        ],
        milestones=[5]
    )

    loss_fn = nn.MSELoss()

    scaler = GradScaler(
        enabled=device.type == "cuda"
    )

    save_dir = f"outputs/{model_name}"

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    best_valid_loss = float("inf")

    for epoch in range(config["epochs"]):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        train_loss = 0.0
        train_steps = 0

        for (u, y, _) in train_data:

            u = u.float()
            y = y.float()

            if (
                y.ndim == 3
                and y.shape[-1] == 1
            ):
                y = y.squeeze(-1)

            if u.ndim == 2:
                u = u.unsqueeze(0)

            if y.ndim == 1:
                y = y.unsqueeze(0)

            # ----------------------------------------------
            # Teacher forcing input
            # y_prev[t] = y[t-1]
            # ----------------------------------------------

            y_prev = torch.zeros_like(y)

            y_prev[:, 1:] = y[:, :-1]

            u = u.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_prev = y_prev.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with autocast(
                device_type=device.type,
                enabled=device.type == "cuda"
            ):

                pred = model(
                    u,
                    y_prev
                )

                loss = loss_fn(
                    pred,
                    y
                )

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_steps += 1

        train_loss /= max(
            train_steps,
            1
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        valid_loss = 0.0
        valid_steps = 0

        with torch.no_grad():

            for (u, y, _) in valid_data:

                u = u.float()
                y = y.float()

                if (
                    y.ndim == 3
                    and y.shape[-1] == 1
                ):
                    y = y.squeeze(-1)

                if u.ndim == 2:
                    u = u.unsqueeze(0)

                if y.ndim == 1:
                    y = y.unsqueeze(0)

                y_prev = torch.zeros_like(y)

                y_prev[:, 1:] = y[:, :-1]

                u = u.to(
                    device,
                    non_blocking=True
                )

                y = y.to(
                    device,
                    non_blocking=True
                )

                y_prev = y_prev.to(
                    device,
                    non_blocking=True
                )

                with autocast(
                    device_type=device.type,
                    enabled=device.type == "cuda"
                ):

                    pred = model(
                        u,
                        y_prev
                    )

                    loss = loss_fn(
                        pred,
                        y
                    )

                valid_loss += loss.item()
                valid_steps += 1

        valid_loss /= max(
            valid_steps,
            1
        )

        scheduler.step()

        if valid_loss < best_valid_loss:

            best_valid_loss = valid_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "scheduler_state_dict":
                        scheduler.state_dict(),

                    "valid_loss":
                        valid_loss,
                },
                f"{save_dir}/best_model.pt"
            )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch + 1} | "
            f"LR: {current_lr:.6e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Valid Loss: {valid_loss:.6f}"
        )

    return model
