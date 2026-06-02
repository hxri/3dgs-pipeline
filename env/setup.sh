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

# ── COLMAP: native binary, or a Docker shim (COLMAP_DOCKER=1) ─────────────────
if [[ "${COLMAP_DOCKER:-0}" == "1" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "==> COLMAP_DOCKER=1 but docker not found. Install Docker (+ nvidia-container-toolkit)."
    exit 1
  fi
  BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
  mkdir -p "$BIN_DIR"
  install -m 0755 env/colmap-docker "$BIN_DIR/colmap"
  echo "==> Installed COLMAP docker shim -> $BIN_DIR/colmap  (image: ${COLMAP_IMAGE:-colmap/colmap:latest})"
  case ":$PATH:" in
    *":$BIN_DIR:"*) : ;;
    *) echo "    NOTE: $BIN_DIR is not on PATH — add this to your shell rc:"
       echo "          export PATH=\"$BIN_DIR:\$PATH\"" ;;
  esac
  echo "    (first colmap call will 'docker pull' the image; or pre-pull it now)"
elif ! command -v colmap >/dev/null 2>&1; then
  echo "==> COLMAP not found. Either:"
  echo "    - install native:  sudo apt-get install -y colmap   /   brew install colmap"
  echo "    - or use Docker:   COLMAP_DOCKER=1 bash env/setup.sh"
  exit 1
fi

# ── ffmpeg (native; uv can't install it) ─────────────────────────────────────
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "==> ffmpeg not found.  Linux: sudo apt-get install -y ffmpeg   macOS: brew install ffmpeg"
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
