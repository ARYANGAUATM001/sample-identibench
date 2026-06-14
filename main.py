import argparse
import datetime
import json
import os
import sys


# ============================================================
# Per-run logging (stdlib only, set up BEFORE the heavy imports
# so the log folder + progress appear immediately, and the whole
# run -- including the slow mamba_ssm import -- is captured)
# ============================================================

class _Tee:
    """Write stream that mirrors output to the console and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_run_dir(model_name):
    """Create logs/<date_time>_<model>/ and tee stdout+stderr into it."""

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("logs", f"{ts}_{model_name}")
    os.makedirs(run_dir, exist_ok=True)

    log_file = open(os.path.join(run_dir, "run.log"), "w")

    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    print(f"==== run {ts} | model={model_name} | logging to {run_dir} ====")

    return run_dir


# ============================================================
# MODEL SELECTION
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    default="mamba1",
    choices=[
        "mamba1",
        "mamba2",
        "mamba3"
    ]
)

args = parser.parse_args()

# start logging immediately, before any heavy import
run_dir = setup_run_dir(args.model)


# ============================================================
# HEAVY IMPORTS (captured in the log from here on)
# ============================================================

print("Importing scientific stack (torch, identibench) ...")

import identibench as idb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

# Speed: use TF32 tensor cores for fp32 matmuls and autotune cuDNN kernels
# (big speedup on Ampere GPUs like the RTX 3090, negligible accuracy impact).
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from utils.seed import set_seed  # noqa: E402
from utils.preprocessing import apply_init_window  # noqa: E402

from configs import DEFAULT_CONFIG  # noqa: E402
from model.trainer import train_model  # noqa: E402


# ============================================================
# IMPORT MODEL
# ============================================================

print(
    f"Loading model libraries for {args.model} "
    "(importing mamba_ssm can take ~15s the first time) ..."
)

if args.model == "mamba1":
    from model.mamba1 import Model  # noqa: E402

elif args.model == "mamba2":
    from model.mamba2 import Model  # noqa: E402

elif args.model == "mamba3":
    from model.mamba3 import Model  # noqa: E402

print(f"Libraries loaded for {args.model}.")


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(context):

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(context.seed)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # Spec
    # --------------------------------------------------------

    spec = context.spec

    init_window = getattr(
        spec,
        "init_window",
        0
    )

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    config = DEFAULT_CONFIG.copy()

    config.update(
        context.hyperparameters
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_data = list(
        context.get_train_sequences()
    )

    valid_data = list(
        context.get_valid_sequences()
    )

    # --------------------------------------------------------
    # Infer input dimension safely
    # --------------------------------------------------------

    u0, _, _ = train_data[0]

    if u0.ndim == 1:
        input_dim = 1
    else:
        input_dim = u0.shape[-1]

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    train_data = [

        apply_init_window(
            u,
            y,
            init_window
        ) + (x,)

        for (u, y, x)
        in train_data
    ]

    valid_data = [

        apply_init_window(
            u,
            y,
            init_window
        ) + (x,)

        for (u, y, x)
        in valid_data
    ]

    # --------------------------------------------------------
    # Normalization (z-score) using TRAIN statistics.
    #
    # The raw outputs are in millivolts with a large dynamic
    # range; regressing them directly makes the MSE objective
    # ill-conditioned (huge / exploding gradients -> NaN) and
    # keeps the RMSE orders of magnitude above SOTA. We train
    # the model on standardized signals and denormalize the
    # predictions back to physical units for the benchmark.
    # --------------------------------------------------------

    all_u = np.concatenate(
        [np.asarray(u, dtype=np.float64).reshape(len(u), -1)
         for (u, y, _) in train_data],
        axis=0,
    )
    all_y = np.concatenate(
        [np.asarray(y, dtype=np.float64).reshape(-1, 1)
         for (u, y, _) in train_data],
        axis=0,
    )

    u_mean = all_u.mean(axis=0)
    u_std = all_u.std(axis=0) + 1e-8
    y_mean = float(all_y.mean())
    y_std = float(all_y.std() + 1e-8)

    def _normalize(data):
        out = []
        for (u, y, attrs) in data:
            un = (np.asarray(u, dtype=np.float32).reshape(len(u), -1)
                  - u_mean) / u_std
            yn = (np.asarray(y, dtype=np.float32).reshape(-1, 1)
                  - y_mean) / y_std
            out.append((un.astype(np.float32), yn.astype(np.float32), attrs))
        return out

    train_data = _normalize(train_data)
    valid_data = _normalize(valid_data)

    # tensors for the predictor (de/normalization at inference)
    u_mean_t = torch.tensor(u_mean, dtype=torch.float32, device=device)
    u_std_t = torch.tensor(u_std, dtype=torch.float32, device=device)

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = Model(

        input_dim=input_dim,

        d_model=config.get(
            "hidden_dim",
            128
        ),

        d_state=config.get(
            "d_state",
            64
        ),

        n_layers=config.get(
            "n_layers",
            6
        ),

        num_classes=1,

        use_skip=config.get("use_skip", True),

    ).to(device)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(

        model=model,

        train_data=train_data,

        valid_data=valid_data,

        config=config,

        device=device,

        model_name=args.model
    )

    # --------------------------------------------------------
    # Load BEST checkpoint
    # --------------------------------------------------------

    ckpt_path = (
        f"outputs/{args.model}/best_model.pt"
    )

    if os.path.exists(ckpt_path):

        checkpoint = torch.load(
            ckpt_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    model.eval()

    # ========================================================
    # Predictor
    # ========================================================

    def predictor(
        u_test,
        y_init=None,
        attrs=None
    ):

        model.eval()

        # ----------------------------------------------------
        # Convert + normalize input (train-set statistics)
        # ----------------------------------------------------

        u_test = torch.as_tensor(
            u_test,
            dtype=torch.float32,
            device=device
        )

        # ensure feature dimension
        if u_test.ndim == 1:
            u_test = u_test.unsqueeze(-1)

        u_test = (u_test - u_mean_t) / u_std_t

        with torch.no_grad():

            # ============================================
            # Warmup handling
            #
            # Models are y_t = f(u_1:t) (not autoregressive
            # on y), so for the prediction task we warm up the
            # hidden state with a neutral (mean) input prefix of
            # the same length as the provided initial condition.
            # In normalized space the mean input is zero.
            # For simulation, y_init is empty -> warm_len = 0.
            # ============================================

            if y_init is not None and len(y_init) > 0:

                warm_len = len(y_init)

                u_warm = torch.zeros(
                    (warm_len, u_test.shape[-1]),
                    dtype=torch.float32,
                    device=device
                )

                u_full = torch.cat([u_warm, u_test], dim=0)

            else:

                warm_len = 0
                u_full = u_test

            # ============================================
            # Full-sequence free-run simulation in fp32.
            # (verified to fit in memory for the full WH/Silverbox
            #  test sequences; avoids state-reset artifacts.)
            # ============================================

            out = model(
                u_full.unsqueeze(0).contiguous()
            )

            if out.ndim == 3 and out.shape[-1] == 1:
                out = out.squeeze(-1)

            # normalized prediction, drop warmup prefix
            y_norm = out.squeeze(0)[warm_len:]

            # ============================================
            # Denormalize back to physical units (mV)
            # ============================================

            y_pred = y_norm * y_std + y_mean

        return (

            y_pred
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1, 1)

        )

    return predictor


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # run_dir / logging already set up at import time (top of file)

    # --------------------------------------------------------
    # Benchmarks
    # --------------------------------------------------------

    # Two datasets only: Silverbox + Wiener-Hammerstein
    # (simulation / free-run task, the headline metric on the
    #  nonlinear-benchmark results-reporting page)
    sim_specs = {

        "Silverbox_Sim":
            idb.simulation_benchmarks[
                "Silverbox_Sim"
            ],

        "WH_Sim":
            idb.simulation_benchmarks[
                "WH_Sim"
            ],
    }

    all_specs = {
        **sim_specs,
    }

    # Optional subset for quick smoke tests, e.g. IDB_BENCH=Silverbox_Sim
    _only = os.environ.get("IDB_BENCH", "").strip()
    if _only:
        keep = {k.strip() for k in _only.split(",")}
        all_specs = {k: v for k, v in all_specs.items() if k in keep}

    # --------------------------------------------------------
    # Run benchmarks
    # --------------------------------------------------------

    n_times = int(os.environ.get("IDB_N_TIMES", "2"))

    hyperparameters = {

        "hidden_dim": int(os.environ.get("IDB_HIDDEN", "128")),

        "d_state": int(os.environ.get("IDB_D_STATE", "64")),

        "n_layers": int(os.environ.get("IDB_N_LAYERS", "6")),

        # Linear u->y feed-through. Helps the feed-forward WH, but is wrong
        # physics for a resonator (Silverbox has no instantaneous feed-through)
        # -> set IDB_SKIP=0 for Silverbox.
        "use_skip": os.environ.get("IDB_SKIP", "1") == "1",

        # Train long enough to actually converge (the few-second
        # runs were badly undertrained). env-overridable.
        "epochs": int(os.environ.get("IDB_EPOCHS", "60")),

        "lr": float(os.environ.get("IDB_LR", "1e-3")),

        # Longer windows + washout so the model learns the
        # dynamics with a warmed-up state instead of cold 100-step
        # snippets.
        "seq_len": int(os.environ.get("IDB_SEQ_LEN", "1024")),

        "washout": int(os.environ.get("IDB_WASHOUT", "100")),

        # Minibatch of random sub-sequence crops -> stable gradients +
        # augmentation. 32 keeps epochs fast (bf16/TF32 then give a net
        # speedup); set IDB_BATCH=64 to trade speed for a bit more accuracy.
        "batch_size": int(os.environ.get("IDB_BATCH", "32")),

        "weight_decay": float(os.environ.get("IDB_WD", "1e-2")),

        "grad_clip": 1.0,

        # Speed: validate (full free-run) every N epochs instead of every one.
        "valid_every": int(os.environ.get("IDB_VALID_EVERY", "5")),

        # Weight EMA: off by default (it lagged and hurt on this short,
        # cosine-annealed training). Enable to experiment, e.g. IDB_EMA=0.99.
        "ema_decay": float(os.environ.get("IDB_EMA", "0")),

        # Speed: bf16 mixed precision on CUDA (verified to work with
        # causal_conv1d, unlike fp16). Set IDB_AMP=none for plain fp32+TF32.
        "amp": os.environ.get("IDB_AMP", "bf16"),

        "compile": False,
    }

    # Save the run configuration into the log folder
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(
            {
                "model": args.model,
                "benchmarks": list(all_specs.keys()),
                "n_times": n_times,
                "hyperparameters": hyperparameters,
            },
            f,
            indent=2,
        )

    results = idb.run_benchmarks(

        all_specs,

        build_model=build_model,

        n_times=n_times,

        hyperparameters=hyperparameters,
    )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    agg = idb.aggregate_benchmark_results(

        results,

        agg_funcs=[
            "mean",
            "std"
        ]
    )

    df = pd.DataFrame(agg)

    # --------------------------------------------------------
    # Persist results
    # --------------------------------------------------------

    # latest-run convenience copy
    out_dir = f"outputs/{args.model}"
    os.makedirs(out_dir, exist_ok=True)

    # write results into BOTH the per-run log folder and outputs/<model>/
    for d in (run_dir, out_dir):
        results.to_csv(os.path.join(d, "raw_results.csv"), index=False)
        df.to_csv(os.path.join(d, "aggregated_results.csv"), index=False)

    print(f"\n===== {args.model} : raw per-run results =====")
    print(results.to_string(index=False))

    print(f"\n===== {args.model} : aggregated (mean / std) =====")
    print(df.to_string(index=False))

    print(f"\nAll logs + results for this run saved to: {run_dir}")
