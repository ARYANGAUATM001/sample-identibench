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
    return nn.functional.mse_loss(pred.float(), target.float())


class _EMA:
    """Exponential moving average of model parameters (improves & stabilizes
    the final model). EMA weights are used for validation and checkpointing."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {
            n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self, model):
        """Load EMA weights into the model; return a backup of the live ones."""
        backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])
        return backup

    @torch.no_grad()
    def swap_out(self, model, backup):
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])


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
    valid_every = max(1, config.get("valid_every", 5))
    # EMA of weights. Off by default: with this short, cosine-annealed
    # training it lags the live model and hurts. Enable with 0<decay<1.
    ema_decay = config.get("ema_decay", 0.0)
    use_ema = 0.0 < ema_decay < 1.0

    # mixed precision: bf16 on CUDA (faster on Ampere, no GradScaler needed,
    # and unlike fp16 it avoids the causal_conv1d stride issue). Configurable.
    amp_mode = config.get("amp", "bf16")
    use_amp = (device.type == "cuda") and (amp_mode in ("bf16", "fp16"))
    amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16

    def autocast_ctx():
        return torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        )

    # --------------------------------------------------------
    # Optimizer / data
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-2),
    )

    save_dir = f"outputs/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    train_seqs = _seq_tensors(train_data, device)
    valid_seqs = _seq_tensors(valid_data, device)

    croppable = [(u, y) for (u, y) in train_seqs if u.shape[0] >= seq_len]
    if not croppable:
        croppable = train_seqs

    total_len = sum(u.shape[0] for (u, _) in croppable)
    steps_per_epoch = max(20, total_len // seq_len)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps_per_epoch,
    )

    ema = _EMA(model, ema_decay) if use_ema else None
    best_valid_loss = float("inf")

    def run_validation():
        model.eval()
        vloss, vcount = 0.0, 0
        with torch.no_grad():
            for (u, y) in valid_seqs:
                with autocast_ctx():
                    pred = model(u.unsqueeze(0))
                if pred.ndim == 3 and pred.shape[-1] == 1:
                    pred = pred.squeeze(-1)
                vloss += _masked_mse(pred, y.unsqueeze(0), washout).item()
                vcount += 1
        return vloss / max(vcount, 1)

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(epochs):

        # ===================== TRAIN ========================
        model.train()
        train_loss = 0.0

        for _ in range(steps_per_epoch):

            bu, by = [], []
            for _ in range(batch_size):
                u, y = random.choice(croppable)
                L = u.shape[0]
                s = random.randint(0, L - seq_len) if L > seq_len else 0
                bu.append(u[s:s + seq_len])
                by.append(y[s:s + seq_len])

            u_batch = torch.stack(bu, dim=0)
            y_batch = torch.stack(by, dim=0)

            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx():
                pred = model(u_batch)
                if pred.ndim == 3 and pred.shape[-1] == 1:
                    pred = pred.squeeze(-1)
                loss = _masked_mse(pred, y_batch, washout)

            if not torch.isfinite(loss):
                print("Invalid loss detected, skipping step")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            scheduler.step()
            if use_ema:
                ema.update(model)

            train_loss += loss.item()

        train_loss /= steps_per_epoch

        # =================== VALIDATION (EMA) ===============
        # validate every `valid_every` epochs (+ always the last one) to save
        # time; evaluate with the EMA weights and checkpoint those.
        is_last = (epoch == epochs - 1)
        if ((epoch + 1) % valid_every == 0) or is_last:

            # evaluate with EMA weights if enabled, else the live model
            backup = ema.swap_in(model) if use_ema else None
            valid_loss = run_validation()

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

            if use_ema:
                ema.swap_out(model, backup)      # restore live weights

            tag = "EMA" if use_ema else "live"
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1} | LR: {current_lr:.6e} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Valid Loss ({tag}): {valid_loss:.6f}"
            )
        else:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1} | LR: {current_lr:.6e} | "
                f"Train Loss: {train_loss:.6f}"
            )

    return model
