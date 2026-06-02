"""Stage 03 — derive 3D instance labels once, so every rendered view gets free masks.

Pipeline:
  (a) SEED  — SAM2 automatic masks + CLIP class assignment on strided training views.
  (b) VOTE  — project Gaussians into seeded views, tally class votes (geometric lift).
  (c) SPLIT — cluster each class spatially (DBSCAN) into instances.

Loads the trained splatfacto scene so Gaussians and cameras share one coordinate frame.
GPU required. See docs/THE_HARD_PART.md for the design rationale.
"""
from __future__ import annotations

import json

import numpy as np

from src.common.cli import common_parser, setup
from src.common import nerfstudio_io as nio
from src.label.lift import assign_classes, cluster_instances, vote_classes
from src.label.sam2_seed import build_amg, ClipClassifier, seed_label_map


def main():
    args = common_parser("Stage 03: 3D instance labeling").parse_args()
    cfg, paths, log = setup(args)
    lc = cfg["label"]
    classes = lc["classes"]
    num_classes = len(classes)

    import cv2
    import torch

    # ── load the trained scene (gaussians + cameras in one frame) ────────────
    config_path = nio.resolve_latest_config(paths.splat)
    log.info("Loading scene: %s", config_path)
    _, pipeline = nio.load_pipeline(config_path)
    model = pipeline.model
    means, *_ = nio.activated_gaussians(model)
    device = means.device
    center, radius = nio.scene_bounds(means)
    log.info("scene: %d gaussians, radius=%.3f", means.shape[0], radius)

    cams, image_paths, filenames = nio.get_train_cameras(pipeline)
    n_cams = len(image_paths)

    # ── (a) SEED on a strided subset of training views ───────────────────────
    amg = build_amg(lc, device)
    clip = ClipClassifier(lc["clip"], classes, device)
    seed_dir = paths.labels / "seed"
    seed_dir.mkdir(parents=True, exist_ok=True)

    stride = max(1, int(lc["seed_stride"]))
    seed_idxs = list(range(0, n_cams, stride))
    log.info("SAM2+CLIP seeding %d/%d views (stride %d)", len(seed_idxs), n_cams, stride)

    seeds = []
    for k, ci in enumerate(seed_idxs):
        img_bgr = cv2.imread(image_paths[ci])
        if img_bgr is None:
            log.warning("could not read %s, skipping", image_paths[ci])
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        label_map = seed_label_map(img_rgb, amg, clip)
        np.save(seed_dir / f"{filenames[ci]}.npy", label_map)
        fx, fy, cx, cy, W, H = nio.intrinsics(cams, ci)
        c2w = torch.tensor(nio.camera_to_world_4x4(cams, ci), dtype=torch.float32, device=device)
        lm = torch.tensor(label_map, dtype=torch.long, device=device)
        seeds.append((lm, fx, fy, cx, cy, W, H, c2w))
        if (k + 1) % 10 == 0:
            log.info("  seeded %d/%d (last: %d labeled px)",
                     k + 1, len(seed_idxs), int((label_map > 0).sum()))

    if not seeds:
        raise SystemExit("No seed masks produced — check SAM2 checkpoint / images.")

    # ── (b) VOTE: lift 2D semantics onto Gaussians ───────────────────────────
    log.info("Voting classes onto %d gaussians across %d views", means.shape[0], len(seeds))
    votes = vote_classes(means, seeds, num_classes, device, log)
    class_ids = assign_classes(votes, int(lc["vote"]["min_votes"]))
    labeled = int((class_ids > 0).sum())
    log.info("labeled %d/%d gaussians (%.1f%%)", labeled, means.shape[0],
             100.0 * labeled / means.shape[0])

    # ── (c) SPLIT classes into instances by 3D clustering ────────────────────
    ic = lc["instance"]
    eps = float(ic["dbscan_eps_scale"]) * radius
    log.info("Clustering instances (DBSCAN eps=%.4f, min_samples=%d)", eps, ic["dbscan_min_samples"])
    inst_ids, inst_classes = cluster_instances(
        means.detach().cpu().numpy(), class_ids.detach().cpu().numpy(),
        num_classes, eps, int(ic["dbscan_min_samples"]), int(ic["max_points_per_class"]), log)
    log.info("found %d instances", len(inst_classes))

    # ── save the output contract ─────────────────────────────────────────────
    out = paths.labels / "instances"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "gaussian_instance_ids.npy", inst_ids)
    (out / "instance_classes.json").write_text(
        json.dumps({str(k): v for k, v in inst_classes.items()}))
    (paths.labels / "classes.json").write_text(json.dumps(classes))
    log.info("✅ labels -> %s  (%d instances over %d classes)", out, len(inst_classes), num_classes)


if __name__ == "__main__":
    main()
