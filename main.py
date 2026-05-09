import argparse
import identibench as idb
import torch

from utils.seed import set_seed
from utils.preprocessing import apply_init_window
from configs import DEFAULT_CONFIG


# ---------------- MODEL SELECTION ----------------

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
    raise ValueError("Invalid model name")


# ---------------- TRAINING ----------------

def train_model(model, train_data, valid_data, config, device):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"]
    )

    loss_fn = torch.nn.MSELoss()

    seq_len = 100

    for epoch in range(config["epochs"]):

        # ---------- TRAIN ----------

        model.train()

        train_loss = 0.0

        for (u, y, _) in train_data:

            u = torch.tensor(
                u,
                dtype=torch.float32
            )

            y = torch.tensor(
                y,
                dtype=torch.float32
            ).view(-1)

            for i in range(0, len(u), seq_len):

                u_batch = (
                    u[i:i + seq_len]
                    .unsqueeze(0)
                    .to(device)
                )

                y_batch = (
                    y[i:i + seq_len]
                    .unsqueeze(0)
                    .to(device)
                )

                optimizer.zero_grad()

                pred = model(u_batch)

                loss = loss_fn(pred, y_batch)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0
                )

                optimizer.step()

                train_loss += loss.item()

        # ---------- VALID ----------

        model.eval()

        valid_loss = 0.0

        with torch.no_grad():

            for (u, y, _) in valid_data:

                u = torch.tensor(
                    u,
                    dtype=torch.float32
                )

                y = torch.tensor(
                    y,
                    dtype=torch.float32
                ).view(-1)

                u = (
                    u[:seq_len]
                    .unsqueeze(0)
                    .to(device)
                )

                y = (
                    y[:seq_len]
                    .unsqueeze(0)
                    .to(device)
                )

                pred = model(u)

                valid_loss += loss_fn(
                    pred,
                    y
                ).item()

        print(
            f"Epoch {epoch + 1} | "
            f"Train: {train_loss:.4f} | "
            f"Valid: {valid_loss:.4f}"
        )

    return model


# ---------------- BUILD MODEL ----------------

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

    train_data = list(
        context.get_train_sequences()
    )

    valid_data = list(
        context.get_valid_sequences()
    )

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

    # ---------- MODEL ----------

    model = Model(
        d_model=config["hidden_dim"],
        num_classes=1
    ).to(device)

    # ---------- TRAIN ----------

    model = train_model(
        model,
        train_data,
        valid_data,
        config,
        device
    )

    # ---------- PREDICTOR ----------

    def predictor(u_test, y_init=None):

        model.eval()

        u_test = torch.tensor(
            u_test,
            dtype=torch.float32
        ).to(device)

        with torch.no_grad():

            if y_init is not None:

                y_init = torch.tensor(
                    y_init,
                    dtype=torch.float32
                ).to(device)

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

                y_full = model(
                    u_full.unsqueeze(0)
                )[0]

                y_pred = y_full[len(y_init):]

            else:

                y_pred = model(
                    u_test.unsqueeze(0)
                )[0]

        return y_pred.reshape(-1, 1)

    return predictor


# ---------------- MAIN ----------------

if __name__ == "__main__":

    sim_specs = {

        "WH_Sim":
            idb.simulation_benchmarks["WH_Sim"],

        "Silverbox_Sim":
            idb.simulation_benchmarks["Silverbox_Sim"],
    }

    pred_specs = {

        "WH_Pred":
            idb.prediction_benchmarks["WH_Pred"],
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

            "hidden_dim": 16,

            "epochs": 3,

            "lr": 1e-3,
        }
    )

    import pandas as pd

    agg = idb.aggregate_benchmark_results(
        results,
        agg_funcs=["mean", "std"]
    )

    df = pd.DataFrame(agg)

    print(df.to_string(index=False))