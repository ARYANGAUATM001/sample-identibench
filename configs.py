# Baseline defaults. The values actually used for a run come from the
# `hyperparameters` dict passed to `idb.run_benchmarks` in main.py (which
# overrides these and is env-overridable); these are the fallbacks used by
# `config.get(...)` in the trainer.
DEFAULT_CONFIG = {

    # model
    "hidden_dim": 128,
    "d_state": 64,
    "n_layers": 6,

    # training
    "lr": 1e-3,
    "epochs": 80,
    "seq_len": 1024,
    "washout": 100,
    "batch_size": 32,

    # optimization
    "weight_decay": 1e-2,
    "grad_clip": 1.0,
}
