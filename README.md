# System Identification Benchmark using IdentiBench

This project benchmarks Mamba-based sequence models for **nonlinear system
identification** (free-run *simulation*) on two standard
[nonlinear-benchmark](https://www.nonlinearbenchmark.org) datasets —
**Silverbox** and **Wiener–Hammerstein** — using the
[IdentiBench](https://github.com/daniel-om-weber/identibench) framework.

Models compared: **`Mamba`** (mamba1), **`Mamba2`** (mamba2), and a small
from-scratch SSM (mamba3).

---

## 📌 Task

For each benchmark IdentiBench provides train / validation / test splits and a
`build_model(context)` contract. The model is a causal map `y_t = f(u_1:t)`
(no output feedback), evaluated in **free-run simulation** over the full test
sequence. The metric is **RMSE in millivolts (mV)** — lower is better.

| Dataset | Type | train len | test sets |
|---|---|---|---|
| Silverbox | Duffing oscillator w/ feedback | 50 000 | multisine (21 688), arrow_full (40 475), arrow_no_extrapolation (32 000) |
| Wiener–Hammerstein | feed-forward (LTI–static NL–LTI) | 80 000 | test (78 800) |

---

## ⚙️ Methodology

A correct system-ID pipeline matters far more than the model choice here. The
key ingredients:

1. **Standardization (z-score).** `u` and `y` are normalized with **training-set
   statistics**; predictions are denormalized back to mV. Regressing raw
   millivolts directly makes the MSE objective ill-conditioned (exploding
   gradients → NaN) and leaves RMSE orders of magnitude above SOTA.
2. **Minibatched random sub-sequence crops.** Each step samples a batch of
   random windows (`seq_len`) from the training sequence — stable gradient
   estimates and strong augmentation.
3. **Washout.** The first `washout` steps of each window are excluded from the
   loss so the model is not penalized while its state is still cold.
4. **fp32 + gradient clipping.** Training runs in fp32 (the `causal_conv1d`
   fp16 channel-last kernel needs stride alignment these 1-channel signals
   violate); gradients are clipped to stabilize training.
5. **Full-sequence free-run evaluation** in fp32 over the entire test sequence.

Hyperparameters (env-overridable, see `main.py`): `hidden_dim=128`,
`d_state=64`, `n_layers=6`, `seq_len=1024`, `washout=100`, `batch_size=32`,
`lr=1e-3`, `weight_decay=1e-2`, cosine LR, AdamW.

> **GPU note:** `Mamba2`'s SSD / `causal_conv1d` CUDA kernels require an Ampere+
> GPU (they raise `map::at` on Turing, e.g. RTX 2080). Run mamba2 on Ampere
> (e.g. RTX 3090). `headdim` is set so `nheads = expand·d_model/headdim` is a
> multiple of 8 (kernel requirement).

---

## 🧠 Models

- **mamba1 — `Mamba`**: stacked Mamba-1 blocks (PreNorm + residual), official
  `mamba-ssm` CUDA kernels.
- **mamba2 — `Mamba2`**: stacked Mamba-2 (SSD) blocks, official `mamba-ssm`
  kernels (`headdim=32`).
- **mamba3 — custom**: a hand-written, pure-PyTorch SSM with a per-timestep
  sequential scan. There is **no official optimized Mamba-3 release**; this is
  an educational reimplementation and is ~50–100× slower to train.

---

## ▶️ Setup & Running

```bash
pip install -r requirements.txt          # torch, identibench, mamba-ssm, causal-conv1d, transformers

python main.py --model mamba1
python main.py --model mamba2
python main.py --model mamba3
```

Results are written to `outputs/<model>/raw_results.csv` and
`aggregated_results.csv`. Useful env overrides: `IDB_EPOCHS`, `IDB_N_TIMES`,
`IDB_SEQ_LEN`, `IDB_WASHOUT`, `IDB_BATCH`, `IDB_N_LAYERS`, `IDB_BENCH`.

---

## 📊 Results

IdentiBench **simulation** RMSE in **mV** (mean ± std over repeated runs;
lower is better).

| Model | Silverbox_Sim | Wiener–Hammerstein_Sim |
|---|---|---|
| **mamba1** (`Mamba`) | **10.19 ± 0.64** | **4.29 ± 3.11** (best run **2.09**) |
| **mamba2** (`Mamba2`) | **17.08 ± 0.07** | **4.08 ± 0.59** |
| *SOTA (nonlinearbenchmark.org)* | *sub-mV* | *~0.2–0.3 mV* |

Silverbox per-test-set RMSE (mamba1): multisine **10.19**, arrow_full **18.32**,
arrow_no_extrapolation **11.37** mV. (The `arrow` sets probe amplitude
extrapolation and are intentionally harder.)

**mamba3** is not competitive: its pure-Python scan can't be trained to
convergence in a practical budget, and it underfits. It is reported as
experimental rather than head-to-head.

---

## 🧠 Interpretation

- **Wiener–Hammerstein (feed-forward)** reaches **~2–4 mV** — normal
  deep-learning territory, within ~7–10× of the ~0.3 mV SOTA.
- **Silverbox** stays around **~10 mV**. It is a *feedback* (Duffing) system;
  a generic feed-forward `u→y` sequence model hits a free-run accuracy floor
  there (confirmed: longer windows / more epochs do not break below it). Its
  sub-mV SOTA comes from tailored physical / state-space models — a modeling
  change, not a pipeline fix.
- The dominant lever throughout was **data normalization + training to
  convergence**, not the specific Mamba variant.

---

## 🎯 Conclusion

With a correct pipeline (normalization, proper training, full-sequence free-run
evaluation), Mamba-based models give sensible, NaN-free simulation results on
these benchmarks — strong on Wiener–Hammerstein and reasonable on Silverbox.
Reaching the published sub-mV SOTA would require benchmark-specific modeling
beyond a generic sequence model.
