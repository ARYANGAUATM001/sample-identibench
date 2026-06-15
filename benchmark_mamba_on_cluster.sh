#!/usr/bin/env bash
#
# Benchmark the three Mamba models (mamba1, mamba2, mamba3) on the GPU cluster.
#
#   ./benchmark_mamba_on_cluster.sh          # default: mamba1/2 ~full, mamba3 reduced
#   ./benchmark_mamba_on_cluster.sh fast     # quick end-to-end smoke (few epochs)
#
# Workflow: sync this folder to the cluster -> run each model on the RTX 3090
# with live output -> pull THAT model's logs back into ./logs/ before the next
# one starts (so you get mamba1/mamba2 results without waiting for slow mamba3).
#
# NOTE: the GPU cluster is only reachable from the UNIVERSITY NETWORK. Connect
# to the campus VPN first, otherwise the SSH / sync steps time out.
#
# Auth: SSH key (no password). The cluster login (user@host) is read from .env
# so no personal identifiers are hard-coded here. Requirements on THIS machine:
# rsync + the SSH private key below.

cd "$(dirname "$0")"

# ---- cluster login (user@host) from .env --------------------------------
if [ ! -f .env ]; then
  echo "ERROR: .env not found (needs a line: username=<user>@<host>)"; exit 1
fi
HOST=$(grep -i '^username' .env | head -1 | sed -E 's/^username[[:space:]]*[:=][[:space:]]*//' | tr -d '[:space:]')
if [ -z "$HOST" ]; then
  echo "ERROR: could not read username=<user>@<host> from .env"; exit 1
fi

KEY="$HOME/.ssh/l3s_purushottam_nawale_id_ed25519"
PROXY="http://web-proxy.rrzn.uni-hannover.de:3128/"
if [ ! -f "$KEY" ]; then echo "ERROR: SSH key not found: $KEY"; exit 1; fi

SSH_OPTS="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30"
SSH="ssh $SSH_OPTS"

# ---- config (default vs fast) -------------------------------------------
if [ "${1:-}" = "fast" ]; then
  M12_ENV="IDB_N_TIMES=1 IDB_EPOCHS=15"
  # mamba3 is a pure-Python scan (~8-12 min/epoch) -> keep it tiny.
  M3_ENV="IDB_N_TIMES=1 IDB_EPOCHS=2 IDB_N_LAYERS=1 IDB_SEQ_LEN=512"
  echo ">>> FAST mode (quick smoke run)"
else
  M12_ENV="IDB_N_TIMES=2 IDB_EPOCHS=80"
  # mamba3 full-budget would take ~5h for a non-competitive (~40 mV) result;
  # keep it short so the run finishes in ~30-40 min. It is experimental.
  M3_ENV="IDB_N_TIMES=1 IDB_EPOCHS=4 IDB_N_LAYERS=1 IDB_SEQ_LEN=512"
  echo ">>> DEFAULT mode (mamba1/2 ~full, mamba3 short+experimental; it is slow)"
fi

# ---- connectivity pre-check (helpful VPN hint) --------------------------
echo ">>> checking cluster connectivity ..."
if ! ssh -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 \
        -o StrictHostKeyChecking=accept-new "$HOST" true 2>/dev/null; then
  echo "ERROR: cannot reach the cluster ($HOST)."
  echo "       The GPU cluster is only reachable from the university network."
  echo "       Connect to the campus VPN and try again."
  exit 1
fi

# ---- sync code to the cluster (once) ------------------------------------
echo ">>> syncing code to cluster ..."
rsync -az -e "ssh $SSH_OPTS" \
  --exclude '.git' --exclude '.venv' --exclude 'logs' --exclude 'outputs' \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '*.pt' --exclude '.env' \
  ./ "$HOST:~/sample-identibench/" || { echo "sync failed"; exit 1; }

# clear any leftover runs once
$SSH "$HOST" "pkill -f 'main.py --model' 2>/dev/null || true"

# ---- run one model, then pull ITS logs back -----------------------------
run_one() {
  local model="$1" envvars="$2"
  echo
  echo "########## [$model] training on the RTX 3090 (live) ##########"
  local REMOTE="export http_proxy=$PROXY https_proxy=$PROXY
source /opt/conda/etc/profile.d/conda.sh && conda activate idbench
cd ~/sample-identibench
export CUDA_VISIBLE_DEVICES=0
$envvars python -u main.py --model $model || echo '!!! $model failed'"
  printf '%s\n' "$REMOTE" | $SSH "$HOST" 'bash -s'

  echo ">>> [$model] syncing its logs to ./logs/ ..."
  rsync -az -e "ssh $SSH_OPTS" "$HOST:~/sample-identibench/logs/" ./logs/ 2>/dev/null
}

run_one mamba1 "$M12_ENV"
run_one mamba2 "$M12_ENV"
run_one mamba3 "$M3_ENV"

echo
echo ">>> DONE.  Per-run folders are in ./logs/<date_time>_<model>/  (run.log, config.json, *_results.csv)"
