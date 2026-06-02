"""Stage 03 — derive 3D instance labels once, so every rendered view gets free masks.

Two sub-steps:
  (a) SEED  — run SAM2 on a strided subset of input frames to get 2D instance masks.
              [implemented as a scaffold: wires up SAM2 + saves masks; fill the model call]
  (b) LIFT  — assign a persistent instance id to each Gaussian by back-projecting the
              seed masks into the 3D scene and voting.
              [TODO: the real research step — see docs/THE_HARD_PART.md]

Output: labels/instances/  (per-Gaussian instance id + {instance_id -> class_id} table)
which the render stage reads to produce per-view masks.

This file is intentionally a runnable scaffold: it validates inputs, sets up the
directory contract the render stage expects, and raises a clear NotImplementedError
at the exact spot you need to implement, rather than failing mysteriously.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.common.cli import common_parser, setup


def seed_masks_with_sam2(frames, lc, out_dir, log):
    """Run SAM2 over strided input frames -> 2D instance masks + class guesses.

    Returns the directory of saved seed masks. Filling in the model call is the
    only TODO here; the I/O contract below is what `lift` consumes.
    """
    try:
        import torch  # noqa: F401
        from sam2.build_sam import build_sam2  # noqa: F401
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "SAM2 not installed. `pip install "
            "git+https://github.com/facebookresearch/sam2.git` and download a "
            f"checkpoint to {lc['sam2_checkpoint']}.\n(import error: {e})"
        )

    ckpt = Path(lc["sam2_checkpoint"])
    if not ckpt.exists():
        raise SystemExit(f"SAM2 checkpoint missing: {ckpt} (see env/setup.sh)")

    seed_frames = frames[:: max(1, lc["seed_stride"])]
    log.info("SAM2 seeding on %d / %d frames", len(seed_frames), len(frames))

    # ------------------------------------------------------------------ TODO
    # 1. predictor = SAM2ImagePredictor(build_sam2(lc["sam2_config"], str(ckpt)))
    # 2. For each seed frame: get masks (automatic mask gen OR prompts), then
    #    assign a class id per mask using your category list lc["classes"]
    #    (e.g. CLIP/text-grounding, or manual prompts for the prototype).
    # 3. Track identities across frames (SAM2 video propagation) so the same
    #    physical object keeps one id — this is what makes the 3D lift clean.
    # 4. Save each mask as out_dir/<frame_stem>/<instance_id>.png plus a
    #    out_dir/seed_meta.json: {frame: {instance_id: class_id}}.
    # ----------------------------------------------------------------------
    raise NotImplementedError(
        "Implement SAM2 seed-mask generation here. The render stage only needs the "
        "directory contract documented above; everything downstream is wired."
    )


def lift_masks_to_3d(paths, seed_dir, lc, log):
    """Back-project 2D seed masks into the 3DGS scene and vote a per-Gaussian id.

    Output contract (consumed by stage 04 render):
        labels/instances/gaussian_instance_ids.npy   int array, len = #gaussians
        labels/instances/instance_classes.json       {instance_id: class_id}
    """
    # ------------------------------------------------------------------ TODO
    # Two viable approaches (pick one for the prototype):
    #  A) Feature-field: add a per-Gaussian instance embedding and optimize it so
    #     rendered instance maps match the seed masks (Gaussian Grouping / SAGA).
    #  B) Geometric vote: for each Gaussian, project its center into every seed
    #     view; the instance whose mask covers it most often wins. Cheaper, no
    #     retraining — a great first cut.
    # See docs/THE_HARD_PART.md for the tradeoffs.
    # ----------------------------------------------------------------------
    raise NotImplementedError(
        "Implement 3D lifting (Gaussian Grouping / SAGA, or geometric voting)."
    )


def main():
    args = common_parser("Stage 03: 3D instance labeling").parse_args()
    cfg, paths, log = setup(args)
    lc = cfg["label"]

    frames = sorted(paths.frames.glob("frame_*.jpg"))
    if not frames:
        raise SystemExit(f"No frames in {paths.frames} — run stage 00 first")

    inst_dir = paths.labels / "instances"
    inst_dir.mkdir(parents=True, exist_ok=True)
    # persist the class list so render/train agree on ids without re-reading config
    (paths.labels / "classes.json").write_text(json.dumps(lc["classes"]))

    seed_dir = paths.labels / "seed"
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_masks_with_sam2(frames, lc, seed_dir, log)   # raises NotImplementedError (scaffold)
    lift_masks_to_3d(paths, seed_dir, lc, log)        # raises NotImplementedError (scaffold)

    log.info("✅ labels -> %s", inst_dir)


if __name__ == "__main__":
    main()
