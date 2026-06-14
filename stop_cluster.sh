#!/usr/bin/env bash
#
# Stop any benchmark runs this project started on the GPU cluster.
#
#   ./stop_cluster.sh
#
# Ctrl-C on benchmark_mamba_on_cluster.sh only stops the LOCAL script; the
# remote `python main.py` keeps using the GPU. This kills it on the cluster.
# Auth via SSH key; cluster login (user@host) read from .env.

cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "ERROR: .env not found (needs a line: username=<user>@<host>)"; exit 1
fi
HOST=$(grep -i '^username' .env | head -1 | sed -E 's/^username[[:space:]]*[:=][[:space:]]*//' | tr -d '[:space:]')
KEY="$HOME/.ssh/tnt_purushottam_nawale_id_ed25519"
SSH_OPTS="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"

echo ">>> stopping benchmark runs on $HOST ..."
ssh $SSH_OPTS "$HOST" 'pkill -9 -f "main.py --model"; sleep 1; echo "remaining main.py procs: $(pgrep -f "main.py --model" | wc -l)"'
echo ">>> done."
