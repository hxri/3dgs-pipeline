"""Stage 03a — turn input frames into 2D semantic label maps.

SAM2 automatic mask generator proposes class-agnostic masks; open_clip assigns each
mask one of your configured classes (or rejects it as background). Output per frame is
an int label map: 0 = background, 1..C = class id + 1.

We classify at the *semantic* level (not instance) on purpose — instance identity is
recovered later in 3D by spatial clustering (see lift.py), which avoids fragile
cross-frame 2D instance tracking.
"""
from __future__ import annotations

import numpy as np


def build_amg(label_cfg, device):
    """SAM2 automatic mask generator."""
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    sam2 = build_sam2(label_cfg["sam2_config"], label_cfg["sam2_checkpoint"], device=device)
    return SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        min_mask_region_area=int(label_cfg["min_mask_area"]),
    )


class ClipClassifier:
    """Zero-shot class assignment for a masked crop. Last text prompt is 'background',
    so confidently-background masks are rejected (returns None)."""

    def __init__(self, clip_cfg, classes, device):
        import open_clip
        import torch

        self.device = device
        self.classes = classes
        self.threshold = float(clip_cfg["score_threshold"])
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_cfg["model"], pretrained=clip_cfg["pretrained"], device=device)
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(clip_cfg["model"])
        prompts = [f"a photo of a {c}" for c in classes] + ["a photo of background"]
        text = tokenizer(prompts).to(device)
        with torch.no_grad():
            tf = self.model.encode_text(text)
            tf = tf / tf.norm(dim=-1, keepdim=True)
        self.text_features = tf

    def classify(self, image_rgb, bbox_xywh):
        """Return a 0-based class index, or None to reject (background / low confidence)."""
        import torch
        from PIL import Image

        x, y, w, h = (int(v) for v in bbox_xywh)
        crop = image_rgb[max(0, y):y + h, max(0, x):x + w]
        if crop.size == 0 or min(crop.shape[:2]) < 4:
            return None
        img = self.preprocess(Image.fromarray(crop)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            f = self.model.encode_image(img)
            f = f / f.norm(dim=-1, keepdim=True)
            probs = (100.0 * f @ self.text_features.T).softmax(-1)[0]
        idx = int(probs.argmax())
        if idx >= len(self.classes):          # background prompt won
            return None
        if float(probs[idx]) < self.threshold:
            return None
        return idx


def seed_label_map(image_rgb: np.ndarray, amg, clip: ClipClassifier) -> np.ndarray:
    """Run SAM2 AMG + CLIP -> HxW semantic label map (0 bg, 1..C class+1).
    Larger masks are painted first so smaller objects win overlaps."""
    masks = amg.generate(image_rgb)
    label = np.zeros(image_rgb.shape[:2], dtype=np.int32)
    for m in sorted(masks, key=lambda d: -d["area"]):
        cls = clip.classify(image_rgb, m["bbox"])
        if cls is None:
            continue
        label[m["segmentation"]] = cls + 1
    return label
