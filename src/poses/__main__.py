"""Stage 01 — recover camera poses from frames/ via COLMAP.

Working. Default backend is `ns-process-data` (nerfstudio's COLMAP wrapper), which
writes a `transforms.json` that stage 02 (splatfacto) consumes directly. A raw COLMAP
backend is also provided if you don't want the nerfstudio dependency here.
"""
from __future__ import annotations

import shutil
import subprocess

from src.common.cli import common_parser, setup


def _run(cmd, log):
    log.info("$ %s", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def ns_process_data(paths, pc, log):
    if shutil.which("ns-process-data") is None:
        raise SystemExit("ns-process-data not found — `pip install nerfstudio` or set "
                         "poses.backend: colmap")
    _run([
        "ns-process-data", "images",
        "--data", str(paths.frames),
        "--output-dir", str(paths.colmap),
        "--matching-method", pc["matcher"],
        "--camera-type", "perspective",
        "--skip-image-processing",  # we already extracted/resized frames
    ], log)


def raw_colmap(paths, pc, log):
    if shutil.which("colmap") is None:
        raise SystemExit("colmap not found on PATH")
    db = paths.colmap / "database.db"
    sparse = paths.colmap / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    _run(["colmap", "feature_extractor", "--database_path", str(db),
          "--image_path", str(paths.frames),
          "--ImageReader.camera_model", pc["camera_model"],
          "--ImageReader.single_camera", "1"], log)
    matcher = {"exhaustive": "exhaustive_matcher",
               "sequential": "sequential_matcher",
               "vocab_tree": "vocab_tree_matcher"}.get(pc["matcher"], "exhaustive_matcher")
    _run(["colmap", matcher, "--database_path", str(db)], log)
    _run(["colmap", "mapper", "--database_path", str(db),
          "--image_path", str(paths.frames), "--output_path", str(sparse)], log)
    log.info("COLMAP sparse model in %s (convert to transforms.json for splatfacto if needed)", sparse)


def main():
    args = common_parser("Stage 01: camera poses (COLMAP)").parse_args()
    cfg, paths, log = setup(args)
    pc = cfg["poses"]

    n = len(list(paths.frames.glob("frame_*.jpg")))
    if n == 0:
        raise SystemExit(f"No frames in {paths.frames} — run stage 00 first")
    log.info("Estimating poses for %d frames (backend=%s)", n, pc["backend"])

    if pc["backend"] == "ns-process-data":
        ns_process_data(paths, pc, log)
    elif pc["backend"] == "colmap":
        raw_colmap(paths, pc, log)
    else:
        raise SystemExit(f"unknown poses.backend: {pc['backend']!r}")
    log.info("✅ poses -> %s", paths.colmap)


if __name__ == "__main__":
    main()
