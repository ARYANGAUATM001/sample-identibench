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

    # ========================================================
    # Device
    # ========================================================

    model = model.to(device)

    # Optional PyTorch 2.x compile
    if config.get("compile", False):
        model = torch.compile(model)

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-2)
    )

    # ========================================================
    # Scheduler
    # ========================================================

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"]
    )

    # ========================================================
    # Loss
    # ========================================================

    loss_fn = nn.MSELoss()

    # ========================================================
    # Mixed precision
    # ========================================================

    scaler = GradScaler(
        enabled=device.type == "cuda"
    )

    # ========================================================
    # Config
    # ========================================================

    seq_len = config.get("seq_len", 100)

    save_dir = f"outputs/{model_name}"

    os.makedirs(save_dir, exist_ok=True)

    best_valid_loss = float("inf")

    # ========================================================
    # Epoch loop
    # ========================================================

    for epoch in range(config["epochs"]):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        train_loss = 0.0
        train_steps = 0

        for (u, y, _) in train_data:

            # ------------------------------------------------
            # Tensor conversion
            # ------------------------------------------------

            u = u.float()
            y = y.float()

            # ------------------------------------------------
            # Target shape fix
            # ------------------------------------------------

            if y.ndim == 3 and y.shape[-1] == 1:
                y = y.squeeze(-1)

            # ------------------------------------------------
            # Ensure batch dimension
            # ------------------------------------------------

            if u.ndim == 2:
                u = u.unsqueeze(0)

            if y.ndim == 1:
                y = y.unsqueeze(0)

            # ------------------------------------------------
            # Sequence chunking
            # Shape:
            # u = (B, L, D)
            # y = (B, L)
            # ------------------------------------------------

            seq_total = u.shape[1]

            for i in range(0, seq_total, seq_len):

                u_batch = (
                    u[:, i:i + seq_len]
                    .to(device, non_blocking=True)
                )

                y_batch = (
                    y[:, i:i + seq_len]
                    .to(device, non_blocking=True)
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                # --------------------------------------------
                # Forward
                # --------------------------------------------

                with autocast(
                    device_type=device.type,
                    enabled=device.type == "cuda"
                ):

                    pred = model(u_batch)

                    # safe squeeze
                    if pred.ndim == 3 and pred.shape[-1] == 1:
                        pred = pred.squeeze(-1)

                    loss = loss_fn(
                        pred,
                        y_batch
                    )

                # --------------------------------------------
                # NaN / Inf protection
                # --------------------------------------------

                if not torch.isfinite(loss):
                    print("Invalid loss detected")
                    continue

                # --------------------------------------------
                # Backward
                # --------------------------------------------

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                scaler.step(optimizer)

                scaler.update()

                train_loss += loss.item()
                train_steps += 1

        # ----------------------------------------------------
        # Average train loss
        # ----------------------------------------------------

        train_loss /= max(train_steps, 1)

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

                if y.ndim == 3 and y.shape[-1] == 1:
                    y = y.squeeze(-1)

                if u.ndim == 2:
                    u = u.unsqueeze(0)

                if y.ndim == 1:
                    y = y.unsqueeze(0)

                seq_total = u.shape[1]

                # --------------------------------------------
                # Validate ALL chunks
                # --------------------------------------------

                for i in range(0, seq_total, seq_len):

                    u_batch = (
                        u[:, i:i + seq_len]
                        .to(device, non_blocking=True)
                    )

                    y_batch = (
                        y[:, i:i + seq_len]
                        .to(device, non_blocking=True)
                    )

                    with autocast(
                        device_type=device.type,
                        enabled=device.type == "cuda"
                    ):

                        pred = model(u_batch)

                        if (
                            pred.ndim == 3
                            and pred.shape[-1] == 1
                        ):
                            pred = pred.squeeze(-1)

                        loss = loss_fn(
                            pred,
                            y_batch
                        )

                    valid_loss += loss.item()
                    valid_steps += 1

        # ----------------------------------------------------
        # Average validation loss
        # ----------------------------------------------------

        valid_loss /= max(valid_steps, 1)

        # ====================================================
        # Scheduler step
        # ====================================================

        scheduler.step()

        # ====================================================
        # Save best checkpoint
        # ====================================================

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

        # ====================================================
        # Logging
        # ====================================================

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1} | "
            f"LR: {current_lr:.6e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Valid Loss: {valid_loss:.6f}"
        )

    return model
