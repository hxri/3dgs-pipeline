"""Stage 06 — the honest test: mAP on REAL held-out frames. Working.

Runs the synthetic-trained model on hand-labeled real frames and reports COCO mAP.
This is the number that answers the thesis: did the 3DGS data engine work?
"""
from __future__ import annotations

import json
from pathlib import Path

from src.common.cli import common_parser, setup
from src.common.coco_to_yolo import convert


def main():
    args = common_parser("Stage 06: evaluate on real frames").parse_args()
    cfg, paths, log = setup(args)
    ec = cfg["eval"]

    real_imgs = paths.root / ec["real_images"]
    real_ann = paths.root / ec["real_annotations"]
    if not real_ann.exists():
        raise SystemExit(
            f"Real annotations missing: {real_ann}\n"
            "Hand-label ~50-100 real frames of the scene (e.g. with CVAT/Roboflow) and "
            "export COCO instances.json. This is the held-out test set — keep it real."
        )

    weights = paths.models / "synthetic" / "weights" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"Trained weights missing: {weights} — run stage 05 first")

    # build a YOLO val dataset from the real COCO set (val only)
    real_yolo = paths.eval / "real_yolo"
    data_yaml = convert(real_ann, real_imgs, real_yolo, val_split=1.0)

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed")

    log.info("Evaluating %s on REAL frames", weights)
    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), split="val",
                        device=cfg["hardware"]["train_gpu"],
                        project=str(paths.eval), name="real", exist_ok=True)

    summary = {
        "box_mAP50_95": float(getattr(metrics.box, "map", float("nan"))),
        "box_mAP50": float(getattr(metrics.box, "map50", float("nan"))),
        "mask_mAP50_95": float(getattr(metrics.seg, "map", float("nan"))),
        "mask_mAP50": float(getattr(metrics.seg, "map50", float("nan"))),
    }
    (paths.eval / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("✅ REAL-frame results: %s", summary)
    log.info("   compare against a real-trained + augmentation baseline to interpret.")


if __name__ == "__main__":
    main()
