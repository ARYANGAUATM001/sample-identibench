#!/bin/bash
# Environment setup for identibench mamba benchmarks on the GPU cluster.
set -e
echo "=== SETUP START $(date) ==="

# RRZN proxy (non-login shells don't source /etc/profile.d/proxy.sh)
export http_proxy=http://web-proxy.rrzn.uni-hannover.de:3128/
export https_proxy=http://web-proxy.rrzn.uni-hannover.de:3128/
export ftp_proxy=http://web-proxy.rrzn.uni-hannover.de:3128/

source /opt/conda/etc/profile.d/conda.sh

# Fresh dedicated env (avoid package conflicts per .env note)
conda env remove -y -n idbench 2>/dev/null || true
conda create -y -n idbench python=3.10
conda activate idbench

python -m pip install --upgrade pip

# Torch with CUDA 12.1 (driver supports up to 13.2)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Prebuilt CUDA wheels for causal-conv1d + mamba-ssm (avoid source build:
# system nvcc is 11.5, too old to compile against cu12 torch)
pip install "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
pip install "https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# identibench + scientific stack
pip install identibench numpy pandas scipy matplotlib tqdm packaging setuptools

echo "=== VERIFY ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
import mamba_ssm
from mamba_ssm import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
import causal_conv1d
print("mamba_ssm OK", mamba_ssm.__version__ if hasattr(mamba_ssm,'__version__') else '?')
import identibench as idb
print("identibench OK; benchmarks:", "WH_Sim" in idb.simulation_benchmarks, "Silverbox_Sim" in idb.simulation_benchmarks)
PY

echo "=== SETUP DONE $(date) ==="
