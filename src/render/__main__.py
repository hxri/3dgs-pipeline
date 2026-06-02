"""Stage 04 — the data engine. Render novel views with free instance masks -> COCO.

Loads the trained scene + the per-Gaussian instance ids from stage 03, samples a camera
trajectory, renders RGB + instance maps via RenderEngine, filters empty views, and writes
a COCO instances.json (consumed by stage 05). GPU required.
"""
from __future__ import annotations

import json

import numpy as np

from src.common.cameras import make_trajectory
from src.common.cli import common_parser, setup
from src.common.coco import CocoWriter
from src.common import nerfstudio_io as nio
from src.render.engine import RenderEngine


def main():
    args = common_parser("Stage 04: render data engine").parse_args()
    cfg, paths, log = setup(args)
    rc = cfg["render"]

    inst_dir = paths.labels / "instances"
    ids_path = inst_dir / "gaussian_instance_ids.npy"
    if not ids_path.exists():
        raise SystemExit(f"{ids_path} missing — run stage 03 (label) first")
    classes = json.loads((paths.labels / "classes.json").read_text())
    inst_classes = {int(k): int(v) for k, v in
                    json.loads((inst_dir / "instance_classes.json").read_text()).items()}
    instance_ids = np.load(ids_path)

    import torch  # noqa: F401

    config_path = nio.resolve_latest_config(paths.splat)
    log.info("Loading scene: %s", config_path)
    _, pipeline = nio.load_pipeline(config_path)
    engine = RenderEngine(pipeline.model, instance_ids, device=pipeline.model.device
                          if hasattr(pipeline.model, "device") else None)

    cams, image_paths, _ = nio.get_train_cameras(pipeline)
    fx, fy, cx, cy, W, H = nio.intrinsics_scaled(cams, 0, tuple(rc["image_size"]))
    input_c2w = np.stack([nio.camera_to_world_4x4(cams, i) for i in range(len(image_paths))])

    poses = make_trajectory(
        rc["trajectory"], center=engine.scene_center,
        radius=engine.scene_radius * rc["radius_scale"], n=rc["num_views"],
        elevation_deg=tuple(rc["elevation_deg"]), input_c2w=input_c2w)

    img_dir = paths.render / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    writer = CocoWriter(classes)

    import cv2
    kept = 0
    for i, c2w in enumerate(poses):
        rgb, inst_map = engine.render(c2w, W, H, fx, fy, cx, cy,
                                      alpha_thresh=float(rc["instance_alpha_thresh"]))
        present = [iid for iid in np.unique(inst_map)
                   if iid != 0 and int(iid) in inst_classes]
        if len(present) < rc["min_visible_instances"]:
            continue
        fname = f"view_{i:06d}.jpg"
        cv2.imwrite(str(img_dir / fname), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        writer.add_view(fname, H, W, inst_map, inst_classes)
        kept += 1
        if kept % 250 == 0:
            log.info("  rendered %d labeled views (%d/%d poses)", kept, i + 1, len(poses))

    out_json = writer.save(paths.render / "instances.json")
    log.info("✅ %d labeled views -> %s  (%s)", kept, img_dir, out_json)
    if kept == 0:
        log.warning("No views had visible instances — check stage 03 labels and "
                    "render.trajectory / radius_scale.")


if __name__ == "__main__":
    main()
