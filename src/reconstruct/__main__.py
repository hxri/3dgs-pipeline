"""Stage 02 — train a 3DGS scene from posed frames via nerfstudio splatfacto.

Working. Pins reconstruction to hardware.recon_gpu. Optionally exports a .ply.
"""
from __future__ import annotations

import os
import shutil
import subprocess

from src.common.cli import common_parser, setup


def _run(cmd, log, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    log.info("[gpu %s] $ %s", gpu, " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, env=env)


def main():
    args = common_parser("Stage 02: 3DGS reconstruction").parse_args()
    cfg, paths, log = setup(args)
    rc = cfg["reconstruct"]
    gpu = cfg["hardware"]["recon_gpu"]

    if shutil.which("ns-train") is None:
        raise SystemExit("ns-train not found — `pip install nerfstudio`")

    transforms = paths.colmap / "transforms.json"
    data_arg = str(paths.colmap) if transforms.exists() else str(paths.colmap)
    if not transforms.exists():
        log.warning("No transforms.json in %s — make sure stage 01 used the "
                    "ns-process-data backend.", paths.colmap)

    out = paths.splat
    _run([
        "ns-train", rc["method"],
        "--data", data_arg,
        "--output-dir", str(out),
        "--max-num-iterations", str(rc["max_num_iterations"]),
        "--viewer.quit-on-train-completion", "True",
    ], log, gpu)

    # find the produced config.yml (nerfstudio nests output by method/timestamp)
    configs = sorted(out.rglob("config.yml"))
    if not configs:
        raise SystemExit(f"training produced no config.yml under {out}")
    latest = configs[-1]
    # leave a stable pointer for downstream stages
    (out / "LATEST_CONFIG").write_text(str(latest))
    log.info("trained model config: %s", latest)

    if rc.get("export_ply"):
        ply_dir = out / "export"
        _run(["ns-export", "gaussian-splat",
              "--load-config", str(latest),
              "--output-dir", str(ply_dir)], log, gpu)
        log.info("exported .ply -> %s", ply_dir)

    log.info("✅ reconstruction -> %s", out)


if __name__ == "__main__":
    main()
