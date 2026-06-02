"""Stage 05 — train an off-the-shelf segmenter on the synthetic data. Working.

Converts render/instances.json -> YOLO-seg dataset, then fine-tunes Ultralytics.
"""
from __future__ import annotations

from src.common.cli import common_parser, setup
from src.common.coco_to_yolo import convert


def main():
    args = common_parser("Stage 05: train downstream segmenter").parse_args()
    cfg, paths, log = setup(args)
    tc = cfg["train"]
    gpu = cfg["hardware"]["train_gpu"]

    coco_json = paths.render / "instances.json"
    if not coco_json.exists():
        raise SystemExit(f"{coco_json} missing — run stage 04 (render) first")

    yolo_dir = paths.render / "yolo"
    log.info("Converting COCO -> YOLO-seg at %s", yolo_dir)
    data_yaml = convert(coco_json, paths.render / "images", yolo_dir,
                        val_split=tc["val_split"])

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed (`pip install ultralytics`)")

    log.info("Fine-tuning %s on synthetic data (gpu %s)", tc["model"], gpu)
    model = YOLO(tc["model"])
    model.train(
        data=str(data_yaml), epochs=tc["epochs"], imgsz=tc["imgsz"],
        batch=tc["batch"], device=gpu, project=str(paths.models), name="synthetic",
        exist_ok=True,
    )
    log.info("✅ weights -> %s", paths.models / "synthetic" / "weights" / "best.pt")


if __name__ == "__main__":
    main()
