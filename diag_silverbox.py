"""Diagnose the Silverbox free-run failure: train a quick mamba1, then on each
test set report whether the prediction tracks the output or just drifts."""
import os
import numpy as np
import torch

import identibench as idb
from model.trainer import train_model
from model.mamba1 import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spec = idb.simulation_benchmarks["Silverbox_Sim"]
ctx = idb.TrainingContext(spec, {}, seed=0)


def to_uy(data):
    out = []
    for u, y, attrs in data:
        out.append((
            np.asarray(u, dtype=np.float64).reshape(len(u), -1),
            np.asarray(y, dtype=np.float64).reshape(-1, 1),
            attrs,
        ))
    return out


train = to_uy(ctx.get_train_sequences())
valid = to_uy(ctx.get_valid_sequences())

all_u = np.concatenate([u for u, y, a in train], 0)
all_y = np.concatenate([y for u, y, a in train], 0)
u_mean, u_std = all_u.mean(0), all_u.std(0) + 1e-8
y_mean, y_std = float(all_y.mean()), float(all_y.std() + 1e-8)


def norm(data):
    return [(((u - u_mean) / u_std).astype(np.float32),
             ((y - y_mean) / y_std).astype(np.float32), a) for u, y, a in data]


input_dim = train[0][0].shape[-1]
model = Model(input_dim=input_dim, d_model=128, d_state=64,
              n_layers=6, num_classes=1).to(device)
cfg = dict(epochs=int(os.environ.get("DIAG_EPOCHS", "40")), seq_len=1024,
           washout=100, batch_size=32, lr=1e-3, weight_decay=1e-2,
           grad_clip=1.0, valid_every=10, ema_decay=0.0, amp="bf16")
model = train_model(model, norm(train), norm(valid), cfg, device, "diag")
ck = torch.load("outputs/diag/best_model.pt", map_location=device)
model.load_state_dict(ck["model_state_dict"])
model.eval()

u_mean_t = torch.tensor(u_mean, dtype=torch.float32, device=device)
u_std_t = torch.tensor(u_std, dtype=torch.float32, device=device)

print("\n================ SILVERBOX FREE-RUN DIAGNOSIS ================")
print(f"train y: mean={y_mean*1e3:.2f} mV  std={y_std*1e3:.2f} mV")
for i, (u, y, attrs) in enumerate(to_uy(ctx.get_test_sequences())):
    ut = torch.as_tensor(u.astype(np.float32), device=device)
    utn = (ut - u_mean_t) / u_std_t
    with torch.no_grad():
        out = model(utn.unsqueeze(0))
        if out.ndim == 3:
            out = out.squeeze(-1)
    pred = (out.squeeze(0).float().cpu().numpy() * y_std + y_mean)
    true = y.reshape(-1)
    n = min(len(pred), len(true))
    pred, true = pred[:n], true[:n]
    e = (pred - true)[50:]               # drop init transient like identibench
    p, t = pred[50:], true[50:]
    rmse = np.sqrt(np.mean(e ** 2))
    h = len(e) // 2
    rmse_1 = np.sqrt(np.mean(e[:h] ** 2))
    rmse_2 = np.sqrt(np.mean(e[h:] ** 2))
    corr = np.corrcoef(p, t)[0, 1]
    print(f"\n--- test set {i} (len {n}) ---")
    print(f"  RMSE            : {rmse*1e3:8.2f} mV   (mean-predictor = {np.std(t)*1e3:.2f} mV)")
    print(f"  worse-than-mean : {'YES' if rmse > np.std(t) else 'no'}  (RMSE/std_true = {rmse/np.std(t):.2f})")
    print(f"  corr(pred,true) : {corr:8.3f}   (1.0 = perfect tracking, ~0 = not tracking)")
    print(f"  RMSE 1st half   : {rmse_1*1e3:8.2f} mV   2nd half: {rmse_2*1e3:8.2f} mV   (>> => drift/divergence)")
    print(f"  std  pred/true  : {np.std(p)*1e3:8.2f} / {np.std(t)*1e3:.2f} mV   (pred too flat if << true)")
    print(f"  mean pred-true  : {(np.mean(p)-np.mean(t))*1e3:8.2f} mV   (DC offset)")
print("=============================================================")
