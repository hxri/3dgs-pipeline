"""QA the data engine: overlay rendered instance masks + boxes + class names on the
rendered RGB so you can eyeball label quality BEFORE training anything.

The #1 failure mode of this whole project is silently-wrong masks (from floaters or
imperfect lifting) poisoning the detector. Run this right after stage 04:

    PYTHONPATH=. python -m tools.qa_overlay --scene room_a --n 40
    # writes data/<scene>/render/qa/*.jpg

It samples N views from render/instances.json, decodes the RLE masks, and writes
side-by-side-ish overlays. Pure CPU.
"""
from __future__ import annotations

import json
import random

import cv2
import numpy as np

from src.common.cli import common_parser, setup

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


def _color(i: int):
    rng = np.random.default_rng(i + 1)
    return tuple(int(c) for c in rng.integers(60, 256, size=3))


def main():
    parser = common_parser("QA: overlay rendered masks on rendered RGB")
    parser.add_argument("--n", type=int, default=30, help="number of views to sample")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg, paths, log = setup(args)

    if mask_utils is None:
        raise SystemExit("pycocotools required (pip install pycocotools)")

    coco_path = paths.render / "instances.json"
    if not coco_path.exists():
        raise SystemExit(f"{coco_path} missing — run stage 04 (render) first")
    coco = json.loads(coco_path.read_text())

    classes = {c["id"]: c["name"] for c in coco["categories"]}
    img_dir = paths.render / "images"
    qa_dir = paths.render / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    anns_by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    images = coco["images"]
    random.Random(args.seed).shuffle(images)
    sample = images[: args.n]

    n_with_labels = 0
    for img in sample:
        path = img_dir / img["file_name"]
        canvas = cv2.imread(str(path))
        if canvas is None:
            continue
        anns = anns_by_img.get(img["id"], [])
        if anns:
            n_with_labels += 1
        overlay = canvas.copy()
        for a in anns:
            col = _color(a["category_id"])
            m = mask_utils.decode(a["segmentation"]).astype(bool)
            overlay[m] = col
            x, y, w, h = (int(v) for v in a["bbox"])
            cv2.rectangle(canvas, (x, y), (x + w, y + h), col, 2)
            cv2.putText(canvas, classes.get(a["category_id"], "?"), (x, max(0, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        blended = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)
        cv2.imwrite(str(qa_dir / img["file_name"]), blended)

    log.info("✅ wrote %d QA overlays -> %s  (%d/%d had labels)",
             len(sample), qa_dir, n_with_labels, len(sample))
    log.info("   Eyeball these: masks should hug object boundaries with no floaters.")


if __name__ == "__main__":
    main()
