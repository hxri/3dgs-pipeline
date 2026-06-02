#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
run_stage src.poses "Stage 01 — camera poses (COLMAP)"
