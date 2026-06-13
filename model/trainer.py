import os
import random

import numpy as np
import torch
import torch.nn as nn


def _seq_tensors(data, device):
    """List of (u, y) tensors on device. u: (L, D), y: (L,)."""
    out = []
    for (u, y, _) in data:
        ut = torch.as_tensor(np.asarray(u), dtype=torch.float32, device=device)
        yt = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=device)
        if ut.ndim == 1:
            ut = ut.unsqueeze(-1)
        yt = yt.reshape(-1)
        out.append((ut, yt))
    return out


def _masked_mse(pred, target, washout):
    """MSE ignoring the first `washout` (cold-state) steps."""
    if washout > 0 and pred.shape[1] > washout:
        pred = pred[:, washout:]
        target = target[:, washout:]
    return nn.functional.mse_loss(pred, target)


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

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    epochs = config["epochs"]
    seq_len = config.get("seq_len", 1024)
    washout = config.get("washout", 0)
    batch_size = config.get("batch_size", 32)
    grad_clip = config.get("grad_clip", 1.0)

    # --------------------------------------------------------
    # Optimizer / schedule / loss
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-2),
    )

    save_dir = f"outputs/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------------------
    # Materialize sequences on device once
    # --------------------------------------------------------

    train_seqs = _seq_tensors(train_data, device)
    valid_seqs = _seq_tensors(valid_data, device)

    # sequences long enough to crop a training window from
    croppable = [
        (u, y) for (u, y) in train_seqs if u.shape[0] >= seq_len
    ]
    if not croppable:
        # fall back to whatever we have (pad-free: use full seqs)
        croppable = train_seqs

    total_len = sum(u.shape[0] for (u, _) in croppable)

    # An "epoch" = enough random-crop minibatches to cover the data.
    steps_per_epoch = max(20, total_len // seq_len)

    # cosine schedule over the real number of optimizer steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * steps_per_epoch,
    )

    best_valid_loss = float("inf")

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(epochs):

        # ====================================================
        # TRAIN  (minibatches of random sub-sequence crops)
        # ====================================================

        model.train()
        train_loss = 0.0

        for _ in range(steps_per_epoch):

            bu, by = [], []

            for _ in range(batch_size):

                u, y = random.choice(croppable)
                L = u.shape[0]

                if L > seq_len:
                    s = random.randint(0, L - seq_len)
                else:
                    s = 0

                bu.append(u[s:s + seq_len])
                by.append(y[s:s + seq_len])

            u_batch = torch.stack(bu, dim=0)          # (B, seq_len, D)
            y_batch = torch.stack(by, dim=0)          # (B, seq_len)

            optimizer.zero_grad(set_to_none=True)

            pred = model(u_batch)
            if pred.ndim == 3 and pred.shape[-1] == 1:
                pred = pred.squeeze(-1)

            loss = _masked_mse(pred, y_batch, washout)

            if not torch.isfinite(loss):
                print("Invalid loss detected, skipping step")
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= steps_per_epoch

        # ====================================================
        # VALIDATION  (free-run over each full valid sequence)
        # ====================================================

        model.eval()
        valid_loss = 0.0
        valid_count = 0

        with torch.no_grad():

            for (u, y) in valid_seqs:

                pred = model(u.unsqueeze(0))
                if pred.ndim == 3 and pred.shape[-1] == 1:
                    pred = pred.squeeze(-1)

                valid_loss += _masked_mse(
                    pred, y.unsqueeze(0), washout
                ).item()
                valid_count += 1

        valid_loss /= max(valid_count, 1)

        # ====================================================
        # Save best checkpoint (lowest free-run valid loss)
        # ====================================================

        if valid_loss < best_valid_loss:

            best_valid_loss = valid_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "valid_loss": valid_loss,
                },
                f"{save_dir}/best_model.pt",
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
