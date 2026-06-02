"""Convert a COCO instances.json + image dir into an Ultralytics YOLO-seg dataset.

Writes:
    <out>/images/{train,val}/*.jpg
    <out>/labels/{train,val}/*.txt   (class + normalized polygon)
    <out>/data.yaml                  (Ultralytics dataset descriptor)
Returns the path to data.yaml.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None

import cv2


def _rle_to_polygons(seg, h, w):
    """COCO RLE -> list of normalized (x,y) polygons for YOLO-seg."""
    if mask_utils is None:
        raise ImportError("pycocotools required")
    m = mask_utils.decode(seg).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if len(c) < 3:
            continue
        c = c.reshape(-1, 2).astype(np.float64)
        c[:, 0] /= w
        c[:, 1] /= h
        polys.append(c.flatten().tolist())
    return polys


def convert(coco_json: str | Path, image_dir: str | Path, out_dir: str | Path,
            val_split: float = 0.1, seed: int = 0) -> Path:
    coco = json.loads(Path(coco_json).read_text())
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)

    classes = [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]
    anns_by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    images = coco["images"]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(images))
    n_val = max(1, int(len(images) * val_split)) if len(images) > 1 else 0
    val_ids = {images[i]["id"] for i in perm[:n_val]}

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for img in images:
        split = "val" if img["id"] in val_ids else "train"
        stem = Path(img["file_name"]).stem
        src = image_dir / img["file_name"]
        if src.exists():
            shutil.copy(src, out_dir / "images" / split / img["file_name"])
        lines = []
        for a in anns_by_img.get(img["id"], []):
            for poly in _rle_to_polygons(a["segmentation"], img["height"], img["width"]):
                coords = " ".join(f"{v:.6f}" for v in poly)
                lines.append(f"{a['category_id']} {coords}")
        (out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(classes))
    )
    return data_yaml
