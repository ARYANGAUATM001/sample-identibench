#!/usr/bin/env bash
#
# Run all three mamba models on the GPU cluster, from this machine.
#
#   ./run_cluster.sh          # default: mamba1/mamba2 full-ish, mamba3 reduced
#   ./run_cluster.sh fast     # quick end-to-end smoke (few epochs)
#
# It (1) syncs this folder to the cluster, (2) runs mamba1, mamba2, mamba3 on
# the RTX 3090 with live output, (3) pulls the per-run logs back into ./logs/.
#
# Auth is via SSH key (no password). The cluster login (user@host) is read
# from .env so no personal identifiers live in this script. Requirements on
# THIS machine: rsync + the SSH private key below.

cd "$(dirname "$0")"

# Cluster login (user@host) read from .env (git-ignored). .env must contain:
#   username=<user>@<host>        (a ':' separator also works)
if [ ! -f .env ]; then
  echo "ERROR: .env not found (needs a line: username=<user>@<host>)"
  exit 1
fi
HOST=$(grep -i '^username' .env | head -1 | sed -E 's/^username[[:space:]]*[:=][[:space:]]*//' | tr -d '[:space:]')
if [ -z "$HOST" ]; then
  echo "ERROR: could not read username=<user>@<host> from .env"
  exit 1
fi

KEY="$HOME/.ssh/tnt_purushottam_nawale_id_ed25519"
PROXY="http://web-proxy.rrzn.uni-hannover.de:3128/"

if [ ! -f "$KEY" ]; then
  echo "ERROR: SSH key not found: $KEY"
  exit 1
fi

SSH_OPTS="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30"
SSH="ssh $SSH_OPTS"

# ---- config (default vs fast) ----------------------------------------------
if [ "${1:-}" = "fast" ]; then
  M12_ENV="IDB_N_TIMES=1 IDB_EPOCHS=15"
  M3_ENV="IDB_N_TIMES=1 IDB_EPOCHS=5 IDB_N_LAYERS=2 IDB_SEQ_LEN=512"
  echo ">>> FAST mode (quick smoke run)"
else
  M12_ENV="IDB_N_TIMES=2 IDB_EPOCHS=80"
  M3_ENV="IDB_N_TIMES=1 IDB_EPOCHS=15 IDB_N_LAYERS=2 IDB_SEQ_LEN=512"
  echo ">>> DEFAULT mode (mamba1/2 ~full, mamba3 reduced; mamba3 is slow)"
fi

echo ">>> [1/3] syncing code to cluster ..."
rsync -az -e "ssh $SSH_OPTS" \
  --exclude '.git' --exclude '.venv' --exclude 'logs' --exclude 'outputs' \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '*.pt' --exclude '.env' \
  ./ "$HOST:~/sample-identibench/" || { echo "sync failed"; exit 1; }

echo ">>> [2/3] running mamba1 -> mamba2 -> mamba3 on the RTX 3090 (live) ..."
REMOTE="export http_proxy=$PROXY https_proxy=$PROXY
source /opt/conda/etc/profile.d/conda.sh && conda activate idbench
cd ~/sample-identibench
export CUDA_VISIBLE_DEVICES=0
pkill -f 'main.py --model' 2>/dev/null || true
echo '########## RUNNING mamba1 ##########'
$M12_ENV python -u main.py --model mamba1 || echo '!!! mamba1 failed'
echo '########## RUNNING mamba2 ##########'
$M12_ENV python -u main.py --model mamba2 || echo '!!! mamba2 failed'
echo '########## RUNNING mamba3 (slow, reduced) ##########'
$M3_ENV python -u main.py --model mamba3 || echo '!!! mamba3 failed'
echo '########## ALL RUNS DONE ##########'"
printf '%s\n' "$REMOTE" | $SSH "$HOST" 'bash -s'

echo ">>> [3/3] pulling logs back to ./logs/ ..."
rsync -az -e "ssh $SSH_OPTS" "$HOST:~/sample-identibench/logs/" ./logs/ 2>/dev/null

echo ">>> DONE.  Per-run folders are in ./logs/<date_time>_<model>/  (run.log, config.json, *_results.csv)"
