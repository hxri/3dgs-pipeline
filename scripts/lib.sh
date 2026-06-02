#!/usr/bin/env bash
# Shared helpers sourced by every scripts/NN_*.sh.
# Resolves repo root, picks the scene, and runs a stage module consistently.
set -euo pipefail

# repo root = parent of this scripts/ dir, regardless of where you invoke from
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# auto-activate the uv venv if one exists and we're not already in a venv
if [[ -z "${VIRTUAL_ENV:-}" && -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

CONFIG="${CONFIG:-configs/default.yaml}"
SCENE="${SCENE:-}"   # empty -> python falls back to $SCENE env or config default

# Pull the default scene out of the config only for log messages
if [[ -z "$SCENE" ]]; then
  SCENE="$(python - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["project"]["scene"])
PY
)"
fi
export SCENE CONFIG

log() { printf '\033[1;36m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

run_stage() {  # run_stage <module> "<human name>"
  local module="$1" name="$2"
  log "▶ $name   (scene=$SCENE)"
  python -m "$module" --scene "$SCENE" --config "$CONFIG"
  log "✔ $name done"
}
