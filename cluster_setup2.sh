
#!/bin/bash
# Continuation: install mamba-ssm + causal-conv1d prebuilt wheels + deps
# (torch 2.4.0+cu121 already installed in env idbench)
set -e
echo "=== SETUP2 START $(date) ==="

export http_proxy=http://web-proxy.rrzn.uni-hannover.de:3128/
export https_proxy=http://web-proxy.rrzn.uni-hannover.de:3128/
export ftp_proxy=http://web-proxy.rrzn.uni-hannover.de:3128/

source /opt/conda/etc/profile.d/conda.sh
conda activate idbench

BASE="https://github.com"
CC="$BASE/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
MS="$BASE/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

pip install "$CC"
pip install "$MS"
pip install identibench numpy pandas scipy matplotlib tqdm packaging setuptools

echo "=== VERIFY ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
from mamba_ssm import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
import causal_conv1d
print("mamba_ssm + causal_conv1d import OK")
import identibench as idb
print("identibench OK; has benchmarks:",
      "WH_Sim" in idb.simulation_benchmarks,
      "Silverbox_Sim" in idb.simulation_benchmarks)
PY

echo "=== SETUP2 DONE $(date) ==="
