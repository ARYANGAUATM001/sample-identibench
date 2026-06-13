import argparse
import os

import identibench as idb
import pandas as pd
import torch

from torch.amp import autocast

from utils.seed import set_seed
from utils.preprocessing import apply_init_window

from configs import DEFAULT_CONFIG
from model.trainer import train_model


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


# ============================================================
# IMPORT MODEL
# ============================================================

if args.model == "mamba1":

    from model.mamba1 import Model

elif args.model == "mamba2":

    from model.mamba2 import Model

elif args.model == "mamba3":

    from model.mamba3 import Model


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

        num_classes=1

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
        # Convert input
        # ----------------------------------------------------

        u_test = torch.as_tensor(
            u_test,
            dtype=torch.float32,
            device=device
        )

        # ensure feature dimension
        if u_test.ndim == 1:
            u_test = u_test.unsqueeze(-1)

        with torch.no_grad():

            # ============================================
            # Warmup handling
            #
            # Models are y_t = f(u_1:t) (not autoregressive
            # on y), so for the prediction task we warm up the
            # hidden state with a zero-input prefix of the same
            # length as the provided initial condition.
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
            # Inference in fp32 over bounded contiguous chunks.
            #
            # causal_conv1d's fp16 channel-last kernel requires
            # 8-element stride alignment that long / autocast
            # inputs violate (raising a stride error on the full
            # test sequences). Running fp32 on contiguous chunks
            # avoids it and bounds memory.
            # ============================================

            chunk = 4096
            L = u_full.shape[0]
            preds = []

            for i in range(0, L, chunk):

                ui = (
                    u_full[i:i + chunk]
                    .unsqueeze(0)
                    .contiguous()
                )

                oi = model(ui)

                if oi.ndim == 3 and oi.shape[-1] == 1:
                    oi = oi.squeeze(-1)

                preds.append(oi.squeeze(0))

            pred = torch.cat(preds, dim=0)

            y_pred = pred[warm_len:]

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

    results = idb.run_benchmarks(

        all_specs,

        build_model=build_model,

        # repeated runs for mean +/- std (env-overridable)
        n_times=int(os.environ.get("IDB_N_TIMES", "2")),

        hyperparameters={

            "hidden_dim": 128,

            "d_state": 64,

            "n_layers": int(os.environ.get("IDB_N_LAYERS", "6")),

            "epochs": int(os.environ.get("IDB_EPOCHS", "10")),

            "lr": 1e-3,

            "seq_len": 100,

            "weight_decay": 1e-2,

            "compile": False,
        }
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

    out_dir = f"outputs/{args.model}"
    os.makedirs(out_dir, exist_ok=True)

    results.to_csv(
        f"{out_dir}/raw_results.csv",
        index=False
    )

    df.to_csv(
        f"{out_dir}/aggregated_results.csv",
        index=False
    )

    print(f"\n===== {args.model} : raw per-run results =====")
    print(results.to_string(index=False))

    print(f"\n===== {args.model} : aggregated (mean / std) =====")
    print(df.to_string(index=False))
