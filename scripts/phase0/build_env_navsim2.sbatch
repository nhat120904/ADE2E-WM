#!/bin/bash
#SBATCH --job-name=p0-env
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Build envs/navsim2: Python 3.9 + NAVSIM v2.2 (+nuplan-devkit v1.2) + DrivoR inference deps.
set -euo pipefail
P=/home/nhatnc129/nhat.nc/ade2e-wm
UV=/home/nhatnc129/.local/bin/uv
export UV_CACHE_DIR=$P/envs/.uv_cache
cd $P
$UV venv --python 3.9 --seed envs/navsim2
source envs/navsim2/bin/activate
$UV pip install --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r third_party/navsim_v2/requirements.txt einops safetensors \
    --override envs/navsim2_overrides.txt
$UV pip install --no-deps -e third_party/navsim_v2
$UV pip freeze > envs/navsim2_freeze.txt
python - <<'PY'
import torch, numpy, timm, navsim, nuplan, pytorch_lightning, hydra
print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("numpy", numpy.__version__, "timm", timm.__version__, "pl", pytorch_lightning.__version__, "hydra", hydra.__version__)
print("navsim", navsim.__file__); print("nuplan", nuplan.__file__)
PY
echo ENV_OK
