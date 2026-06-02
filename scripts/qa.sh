#!/usr/bin/env bash
# Eyeball rendered mask quality before training:  SCENE=room_a bash scripts/qa.sh [N]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
N="${1:-30}"
log "▶ QA overlays (scene=$SCENE, n=$N)"
python -m tools.qa_overlay --scene "$SCENE" --config "$CONFIG" --n "$N"
