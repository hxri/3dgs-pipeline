"""Stage 00 — extract frames from a video (or copy an image folder) into frames/.

Working. Uses ffmpeg for video; for an image folder it just resizes + copies.
"""
from __future__ import annotations

import glob
import shutil
import subprocess
from pathlib import Path

import cv2

from src.common.cli import common_parser, setup

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _resize_inplace(path: Path, long_edge: int):
    if long_edge <= 0:
        return
    img = cv2.imread(str(path))
    if img is None:
        return
    h, w = img.shape[:2]
    scale = long_edge / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(path), img)


def main():
    args = common_parser("Stage 00: extract frames").parse_args()
    cfg, paths, log = setup(args)
    fc = cfg["frames"]

    matches = sorted(glob.glob(str(paths.root / fc["video_glob"])))
    if not matches:
        # maybe they pointed at a folder of images
        matches = [str(p) for p in sorted(paths.raw.glob("*")) if p.suffix.lower() in IMG_EXTS]
    if not matches:
        raise SystemExit(f"No input found under {paths.root} matching {fc['video_glob']!r} "
                         f"or images in {paths.raw}")

    out = paths.frames
    for f in out.glob("*.jpg"):
        f.unlink()

    first = Path(matches[0])
    if first.suffix.lower() in VIDEO_EXTS:
        log.info("Extracting frames from video %s @ %s fps (cap %d)",
                 first.name, fc["fps"], fc["max_frames"])
        # sample at fps, then we trim to max_frames
        cmd = [
            "ffmpeg", "-y", "-i", str(first),
            "-vf", f"fps={fc['fps']}",
            "-q:v", "2",
            str(out / "frame_%05d.jpg"),
        ]
        subprocess.run(cmd, check=True)
        frames = sorted(out.glob("frame_*.jpg"))
        if len(frames) > fc["max_frames"]:
            stride = len(frames) / fc["max_frames"]
            keep = {frames[round(i * stride)] for i in range(fc["max_frames"])}
            for fr in frames:
                if fr not in keep:
                    fr.unlink()
    else:
        log.info("Copying %d images", len(matches))
        for i, m in enumerate(matches[: fc["max_frames"]]):
            shutil.copy(m, out / f"frame_{i:05d}.jpg")

    kept = sorted(out.glob("frame_*.jpg"))
    for fr in kept:
        _resize_inplace(fr, fc["resize_long_edge"])
    log.info("✅ %d frames -> %s", len(kept), out)


if __name__ == "__main__":
    main()
