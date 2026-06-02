"""Stage 04 — the data engine. Render novel views with free instance masks -> COCO.

Real & reusable here: trajectory sampling (src.common.cameras), the per-view loop,
visibility filtering, and COCO export (src.common.coco).
TODO: the actual 3DGS forward pass that turns a camera pose + the per-Gaussian
instance ids (from stage 03) into (rgb, instance_map). That's the one model-specific
hook — everything around it is done.

Output: render/images/*.jpg  +  render/instances.json  (COCO), ready for stage 05.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.common.cameras import make_trajectory
from src.common.cli import common_parser, setup
from src.common.coco import CocoWriter


def load_scene(paths, gpu, log):
    """Load the trained splatfacto model + per-Gaussian instance ids from stage 03.

    Returns an object exposing:
        .scene_center -> (3,)         scene centroid (for orbit target)
        .scene_radius -> float        bounding radius
        .input_c2w    -> (N,4,4)      input camera poses (for jitter_input traj)
        .render(c2w, W, H) -> (rgb uint8 HxWx3, instance_map int HxW)
    """
    ptr = paths.splat / "LATEST_CONFIG"
    if not ptr.exists():
        raise SystemExit(f"{ptr} missing — run stage 02 (reconstruct) first")
    inst_ids = paths.labels / "instances" / "gaussian_instance_ids.npy"
    if not inst_ids.exists():
        raise SystemExit(f"{inst_ids} missing — run stage 03 (label) first")

    # ------------------------------------------------------------------ TODO
    # 1. Load splatfacto pipeline from ptr.read_text() via nerfstudio's
    #    eval_setup(config_path); grab the GaussianSplattingModel.
    # 2. Load per-Gaussian instance ids (np.load(inst_ids)) and attach as an
    #    extra channel you can rasterize alongside RGB (render the argmax/winner
    #    per pixel to get instance_map). gsplat lets you rasterize arbitrary
    #    per-Gaussian features — render the instance-id channel with nearest /
    #    max-alpha compositing so masks stay crisp.
    # 3. Compute scene_center / scene_radius from gaussian means.
    # ----------------------------------------------------------------------
    raise NotImplementedError(
        "Implement 3DGS scene loading + instance-aware render(). See "
        "docs/THE_HARD_PART.md. Everything downstream of render() already works."
    )


def main():
    args = common_parser("Stage 04: render data engine").parse_args()
    cfg, paths, log = setup(args)
    rc = cfg["render"]
    gpu = cfg["hardware"]["recon_gpu"]

    classes = json.loads((paths.labels / "classes.json").read_text())
    inst_classes = {int(k): int(v) for k, v in
                    json.loads((paths.labels / "instances" / "instance_classes.json").read_text()).items()}

    scene = load_scene(paths, gpu, log)  # raises NotImplementedError (scaffold)

    W, H = rc["image_size"]
    poses = make_trajectory(
        rc["trajectory"], center=scene.scene_center,
        radius=scene.scene_radius * rc["radius_scale"], n=rc["num_views"],
        elevation_deg=tuple(rc["elevation_deg"]), input_c2w=scene.input_c2w,
    )

    img_dir = paths.render / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    writer = CocoWriter(classes)

    kept = 0
    for i, c2w in enumerate(poses):
        rgb, instance_map = scene.render(c2w, W, H)
        present = [iid for iid in np.unique(instance_map)
                   if iid != 0 and int(iid) in inst_classes]
        if len(present) < rc["min_visible_instances"]:
            continue
        fname = f"view_{i:06d}.jpg"
        cv2.imwrite(str(img_dir / fname), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        writer.add_view(fname, H, W, instance_map, inst_classes)
        kept += 1
        if kept % 250 == 0:
            log.info("  rendered %d labeled views", kept)

    out_json = writer.save(paths.render / "instances.json")
    log.info("✅ %d labeled views -> %s  (%s)", kept, img_dir, out_json)


if __name__ == "__main__":
    main()
