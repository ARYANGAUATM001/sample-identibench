import argparse
import identibench as idb
import torch
import pandas as pd

from utils.seed import set_seed
from utils.preprocessing import apply_init_window
from configs import DEFAULT_CONFIG
from trainer import train_model


# ================= MODEL SELECTION =================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    default="mamba1"
)

args = parser.parse_args()


if args.model == "mamba1":

    from models.mamba1 import Model

elif args.model == "mamba2":

    from models.mamba2 import Model

elif args.model == "mamba3":

    from models.mamba3 import Model

else:

    raise ValueError(
        "Invalid model name"
    )


# ================= BUILD MODEL =================

def build_model(context):

    set_seed(context.seed)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    spec = context.spec

    init_window = getattr(
        spec,
        "init_window",
        0
    )

    config = DEFAULT_CONFIG.copy()

    config.update(
        context.hyperparameters
    )

    # ---------- LOAD DATA ----------

    train_data = list(
        context.get_train_sequences()
    )

    valid_data = list(
        context.get_valid_sequences()
    )

    # infer input dimension
    u0, _, _ = train_data[0]

    input_dim = u0.shape[1]

    # ---------- PREPROCESS ----------

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

    # ---------- BUILD MODEL ----------

    model = Model(

        input_dim=input_dim,

        d_model=config["hidden_dim"],

        num_classes=1

    ).to(device)

    # ---------- TRAIN ----------

    model = train_model(

        model=model,

        train_data=train_data,

        valid_data=valid_data,

        config=config,

        device=device,

        model_name=args.model
    )

    # ---------- PREDICTOR ----------

    def predictor(
            u_test,
            y_init=None
    ):

        model.eval()

        u_test = torch.tensor(
            u_test,
            dtype=torch.float32
        ).to(device)

        with torch.no_grad():

            # warmup sequence
            if y_init is not None:

                y_init = torch.tensor(
                    y_init,
                    dtype=torch.float32
                ).to(device)

                # dummy warmup inputs
                u_warm = torch.zeros(

                    (
                        len(y_init),
                        u_test.shape[-1]
                    ),

                    device=device
                )

                u_full = torch.cat(
                    [u_warm, u_test],
                    dim=0
                )

                pred = model(
                    u_full.unsqueeze(0)
                )

                pred = pred.squeeze(0)

                y_pred = pred[
                    len(y_init):
                ]

            else:

                pred = model(
                    u_test.unsqueeze(0)
                )

                y_pred = pred.squeeze(0)

        return (
            y_pred
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, 1)
        )

    return predictor


# ================= MAIN =================

if __name__ == "__main__":

    sim_specs = {

        "WH_Sim":
            idb.simulation_benchmarks[
                "WH_Sim"
            ],

        "Silverbox_Sim":
            idb.simulation_benchmarks[
                "Silverbox_Sim"
            ],
    }

    pred_specs = {

        "WH_Pred":
            idb.prediction_benchmarks[
                "WH_Pred"
            ],
    }

    all_specs = {
        **sim_specs,
        **pred_specs
    }

    results = idb.run_benchmarks(

        all_specs,

        build_model=build_model,

        n_times=2,

        hyperparameters={

            "hidden_dim": 128,

            "epochs": 10,

            "lr": 1e-3,

            "seq_len": 100,
        }
    )

    agg = idb.aggregate_benchmark_results(

        results,

        agg_funcs=[
            "mean",
            "std"
        ]
    )

    df = pd.DataFrame(agg)

    print(
        df.to_string(index=False)
    )
