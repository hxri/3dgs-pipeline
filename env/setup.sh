#!/usr/bin/env bash
# One-time environment setup for the 3dgs-data-engine.
# Creates a conda env and installs the GPU stacks with CUDA-matched wheels.
#
# Usage:  bash env/setup.sh
# Then:   conda activate 3dgs-engine
set -euo pipefail

ENV_NAME="${ENV_NAME:-3dgs-engine}"
PY_VER="${PY_VER:-3.10}"
CUDA_TAG="${CUDA_TAG:-cu124}"      # match your driver: cu121 / cu124 ...

echo "==> Creating conda env '$ENV_NAME' (python $PY_VER)"
conda create -y -n "$ENV_NAME" python="$PY_VER"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "==> Installing system-ish deps (COLMAP via conda-forge)"
conda install -y -c conda-forge colmap ffmpeg

echo "==> PyTorch ($CUDA_TAG)"
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "==> Project python deps"
pip install -r requirements.txt

echo "==> Nerfstudio + gsplat (3DGS reconstruction & rendering)"
pip install nerfstudio        # pulls a compatible gsplat

echo "==> SAM2 (seed masks) + open_clip (class assignment)"
pip install "git+https://github.com/facebookresearch/sam2.git"
pip install open_clip_torch
mkdir -p checkpoints
echo "    Download a SAM2 checkpoint into ./checkpoints, e.g.:"
echo "    wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

cat <<EOF

==> Done.
    conda activate $ENV_NAME
    SCENE=room_a bash scripts/run_all.sh

    Sanity check the GPU stack:
      python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"
EOF
