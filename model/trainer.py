import os
import numpy as np
import torch
import torch.nn as nn

from torch.amp import autocast, GradScaler


def _to_batched(u, y):
    """Convert a single (u, y) numpy/tensor pair to batched tensors.

    Returns u of shape (1, L, D) and y of shape (1, L).
    """

    u = torch.as_tensor(np.asarray(u), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(y), dtype=torch.float32)

    # (L,) -> (L, 1)
    if u.ndim == 1:
        u = u.unsqueeze(-1)

    # (L, D) -> (1, L, D)
    u = u.unsqueeze(0)

    # any shape -> (1, L)
    y = y.reshape(-1).unsqueeze(0)

    return u, y


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
    # Precision
    #
    # Train in fp32: causal_conv1d's fp16 channel-last kernel
    # (used by Mamba/Mamba2) requires 8-element stride alignment
    # that these single-channel system-ID inputs violate, raising
    # a stride error. fp32 avoids it; the models are small enough
    # that fp32 on the GPU is fast.
    # ========================================================

    use_amp = False

    scaler = GradScaler(
        enabled=use_amp
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
            # Tensor conversion + shape normalization
            # u -> (1, L, D), y -> (1, L)
            # ------------------------------------------------

            u, y = _to_batched(u, y)

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
                    enabled=use_amp
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

                u, y = _to_batched(u, y)

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
                        enabled=use_amp
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
