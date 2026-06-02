#!/usr/bin/env bash
# Run the full pipeline end-to-end for one scene.
#   SCENE=room_a bash scripts/run_all.sh
#   SCENE=room_a START=03 bash scripts/run_all.sh   # resume from a stage
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

START="${START:-00}"
declare -a STAGES=(
  "00:src.frames:Stage 00 — extract frames"
  "01:src.poses:Stage 01 — camera poses"
  "02:src.reconstruct:Stage 02 — 3DGS reconstruction"
  "03:src.label:Stage 03 — 3D instance labeling"
  "04:src.render:Stage 04 — render data engine"
  "05:src.train:Stage 05 — train segmenter"
  "06:src.eval:Stage 06 — evaluate on real frames"
)

for entry in "${STAGES[@]}"; do
  IFS=":" read -r num module name <<<"$entry"
  [[ "$num" < "$START" ]] && { log "↷ skip $name"; continue; }
  run_stage "$module" "$name"
done
log "✅ pipeline complete for scene=$SCENE"
