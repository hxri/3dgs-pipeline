#!/usr/bin/env bash
# One-time environment setup for the 3dgs-data-engine, using uv.
# Creates a .venv and installs the GPU stacks with CUDA-matched wheels.
#
# Usage:  bash env/setup.sh
# Then:   source .venv/bin/activate
#
# Native deps (COLMAP, ffmpeg) are NOT python packages — install them with your
# system package manager first (see the check below).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY_VER="${PY_VER:-3.10}"
CUDA_TAG="${CUDA_TAG:-cu124}"      # match your driver: cu121 / cu124 ...

# ── uv ───────────────────────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
  echo "==> uv not found. Install it, then re-run:"
  echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "    # or: pipx install uv   /   brew install uv"
  exit 1
fi

# ── native binaries (uv can't install these) ─────────────────────────────────
missing=()
command -v colmap  >/dev/null 2>&1 || missing+=(colmap)
command -v ffmpeg  >/dev/null 2>&1 || missing+=(ffmpeg)
if [ "${#missing[@]}" -gt 0 ]; then
  echo "==> Missing native deps: ${missing[*]}"
  echo "    Linux:  sudo apt-get install -y colmap ffmpeg"
  echo "    macOS:  brew install colmap ffmpeg"
  echo "    (COLMAP needs a CUDA-enabled build for fast feature matching on GPU.)"
  echo "    Install them, then re-run this script."
  exit 1
fi

# ── python env ───────────────────────────────────────────────────────────────
echo "==> Creating .venv (python $PY_VER) with uv"
uv venv --python "$PY_VER"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> PyTorch ($CUDA_TAG)"
uv pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "==> Project python deps"
uv pip install -r requirements.txt

echo "==> Nerfstudio + gsplat (3DGS reconstruction & rendering)"
uv pip install nerfstudio        # pulls a compatible gsplat

echo "==> SAM2 (seed masks) + open_clip (class assignment)"
uv pip install "git+https://github.com/facebookresearch/sam2.git"
uv pip install open_clip_torch

mkdir -p checkpoints
echo "    Download a SAM2 checkpoint into ./checkpoints, e.g.:"
echo "    wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

cat <<EOF

==> Done.
    source .venv/bin/activate
    SCENE=room_a bash scripts/run_all.sh

    Sanity check the GPU stack:
      python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"
EOF
