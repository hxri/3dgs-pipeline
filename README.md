# 3dgs-data-engine

Turn a casual room walkthrough into a **3D Gaussian Splatting** scene, then use that
scene as a **data-generation engine**: render thousands of novel views with pixel-exact
**instance masks + 2D boxes** and train an off-the-shelf detector/segmenter on the
synthetic data.

> **Thesis:** *Can a single video of a room replace a hand-labeled
> instance-segmentation dataset?*

## Pipeline

```
00 frames      video  ──ffmpeg──▶            frames/
01 poses       frames ──COLMAP──▶            colmap/  (transforms.json + sparse)
02 reconstruct poses  ──gsplat──▶            splat/   (trained 3DGS .ply)
03 label       frames ──SAM2 + lift──▶       labels/  (per-Gaussian instance ids)
04 render      splat  ──novel views──▶       render/  (RGB + masks + COCO json)
05 train       render ──YOLO/Detectron2──▶   models/  (detector weights)
06 eval        real   ──mAP──▶               eval/    (synthetic vs real baseline)
```

Each stage is:
- a **shell script** in `scripts/NN_<stage>.sh` (the thing you run), and
- a **Python module** in `src/<stage>/` (the thing it calls).

Stages read one config (`configs/default.yaml`) and a `SCENE` name, and write into
`data/<scene>/<stage>/`. Stages are independent — re-run any one without redoing the rest.

## Quickstart

```bash
# 0. one-time environment setup (uv venv + gsplat + sam2 + ultralytics)
#    (install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh)
#    (COLMAP + ffmpeg are native deps — apt/brew install them; setup.sh checks)
bash env/setup.sh
source .venv/bin/activate            # the stage scripts also auto-activate it

# 1. drop a capture in place
mkdir -p data/room_a/raw && cp ~/room_a.mp4 data/room_a/raw/

# 2. run a stage (SCENE defaults to the one in configs/default.yaml)
SCENE=room_a bash scripts/00_frames.sh
SCENE=room_a bash scripts/01_poses.sh
SCENE=room_a bash scripts/02_reconstruct.sh

# ... or run the whole thing
SCENE=room_a bash scripts/run_all.sh
```

## Status of each stage

| Stage | Script | Status |
|-------|--------|--------|
| 00 frames      | `00_frames.sh`      | ✅ ffmpeg frame extraction |
| 01 poses       | `01_poses.sh`       | ✅ wraps `ns-process-data` / COLMAP |
| 02 reconstruct | `02_reconstruct.sh` | ✅ wraps `ns-train splatfacto` |
| 03 label       | `03_label.sh`       | ✅ SAM2 + CLIP seeding → geometric voting → DBSCAN instances |
| 04 render      | `04_render.sh`      | ✅ gsplat instance-aware render → COCO |
| 05 train       | `05_train.sh`       | ✅ wraps Ultralytics YOLO-seg |
| 06 eval        | `06_eval.sh`        | ✅ Ultralytics val / pycocotools |

Plus a QA step (run it after stage 04, before training):

```bash
SCENE=room_a bash scripts/qa.sh 40   # overlays -> data/<scene>/render/qa/
```

All seven stages are implemented end-to-end. The GPU stages (02–04) were written against
the nerfstudio / gsplat / SAM2 APIs but need a real GPU + capture to validate — see
`docs/THE_HARD_PART.md` for the design and the known approximations (occlusion in voting,
scene `up` assumption, etc.).

## Hardware

Built for a 2× RTX 6000 Pro (96 GB) box. `configs/default.yaml` pins reconstruction +
the render farm to `GPU 0` and training to `GPU 1` so the loop runs on one machine.

## Layout

```
configs/        pipeline config
env/            environment setup
scripts/        NN_<stage>.sh  +  lib.sh  +  run_all.sh
src/            python modules behind the scripts
  common/       config, logging, camera trajectories, COCO export
  frames/ poses/ reconstruct/ label/ render/ train/ eval/
data/           inputs + all stage outputs (gitignored)
docs/           notes
```
