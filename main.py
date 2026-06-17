import argparse
import os

import identibench as idb
import pandas as pd
import torch

from torch.amp import autocast

from utils.seed import set_seed
from utils.preprocessing import apply_init_window

from configs import DEFAULT_CONFIG
from trainer import train_model




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



if args.model == "mamba1":

    from models.mamba1 import Model

elif args.model == "mamba2":

    from models.mamba2 import Model

elif args.model == "mamba3":

    from models.mamba3 import Model




def build_model(context):

   

    set_seed(context.seed)

    

    device = torch.device(
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

 

    train_data = list(
        context.get_train_sequences()
    )

    valid_data = list(
        context.get_valid_sequences()
    )


    u0, _, _ = train_data[0]

    if u0.ndim == 1:
        input_dim = 1
    else:
        input_dim = u0.shape[-1]

 

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

  

    model = train_model(

        model=model,

        train_data=train_data,

        valid_data=valid_data,

        config=config,

        device=device,

        model_name=args.model
    )

   

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

 

    def predictor(
        u_test,
        y_init=None,
        attrs=None
    ):

        model.eval()

        u_test = torch.as_tensor(
            u_test,
            dtype=torch.float32,
            device=device
        )

        if u_test.ndim == 1:
            u_test = u_test.unsqueeze(-1)

        with torch.no_grad():

            with autocast(
                device_type=device.type,
                enabled=device.type == "cuda"
            ):

                seq_len = u_test.shape[0]

                # ----------------------------------------
                # Initialize from benchmark warmup
                # ----------------------------------------

                if y_init is None:

                    prev_y = torch.zeros(
                        1,
                        dtype=torch.float32,
                        device=device
                    )

                else:

                    y_init = torch.as_tensor(
                        y_init,
                        dtype=torch.float32,
                        device=device
                    ).reshape(-1)

                    prev_y = y_init[-1:]

                predictions = []

               

                for t in range(seq_len):

                    u_t = (
                        u_test[t:t + 1]
                        .unsqueeze(0)
                    )

                    y_prev_t = prev_y.view(
                        1,
                        1
                    )

                    pred_t = model(
                        u_t,
                        y_prev_t
                    )

                    pred_t = pred_t.reshape(-1)

                    predictions.append(
                        pred_t
                    )

                    prev_y = pred_t

                y_pred = torch.cat(
                    predictions,
                    dim=0
                )

        return (
            y_pred
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1, 1)
        )

    return predictor




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

        n_times=5,

        hyperparameters={

            "hidden_dim": 128,

            "d_state": 64,

            "n_layers": 6,

            "epochs": 100,

            "lr": 1e-3,

            "seq_len": 100,

            "weight_decay": 1e-2,

            "compile": False,
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
