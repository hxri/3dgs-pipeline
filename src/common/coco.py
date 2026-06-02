"""Build a COCO instance-segmentation dataset from rendered (rgb, instance_map) pairs.

The render stage produces, per view, an RGB image plus an integer instance map and a
{instance_id -> class_id} table. This module turns those into a standard COCO
`instances.json` (boxes + RLE masks) that Ultralytics / Detectron2 / pycocotools read.

Pure CPU. Uses pycocotools for RLE encoding.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from pycocotools import mask as mask_utils
except ImportError:  # keep import-safe so non-render machines can still import
    mask_utils = None


class CocoWriter:
    def __init__(self, classes: list[str]):
        self.classes = classes
        self.images: list[dict] = []
        self.annotations: list[dict] = []
        self.categories = [{"id": i, "name": c} for i, c in enumerate(classes)]
        self._img_id = 0
        self._ann_id = 0

    def add_view(self, file_name: str, height: int, width: int,
                 instance_map: np.ndarray, inst_to_class: dict[int, int],
                 min_area: int = 1) -> int:
        """Register one rendered view and all its instance masks.

        instance_map: HxW int array, 0 = background, >0 = instance id
        inst_to_class: maps instance id -> class id (index into self.classes)
        Returns the number of annotations added.
        """
        if mask_utils is None:
            raise ImportError("pycocotools is required to encode masks (pip install pycocotools)")

        img_id = self._img_id
        self._img_id += 1
        self.images.append({
            "id": img_id, "file_name": file_name,
            "height": int(height), "width": int(width),
        })

        added = 0
        for inst_id in np.unique(instance_map):
            if inst_id == 0:
                continue
            cls = inst_to_class.get(int(inst_id))
            if cls is None:
                continue
            binary = np.asfortranarray(instance_map == inst_id).astype(np.uint8)
            area = int(binary.sum())
            if area < min_area:
                continue
            rle = mask_utils.encode(binary)
            rle["counts"] = rle["counts"].decode("ascii")  # json-serializable
            x, y, w, h = [float(v) for v in mask_utils.toBbox(rle)]
            self.annotations.append({
                "id": self._ann_id, "image_id": img_id, "category_id": int(cls),
                "segmentation": rle, "bbox": [x, y, w, h], "area": area,
                "iscrowd": 0,
            })
            self._ann_id += 1
            added += 1
        return added

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "images": self.images,
                "annotations": self.annotations,
                "categories": self.categories,
            }, f)
        return path

    def __len__(self) -> int:
        return len(self.images)
