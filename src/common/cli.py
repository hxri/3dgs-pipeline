"""Boilerplate shared by every stage's `__main__`."""
from __future__ import annotations

import argparse

from .config import get_paths, load_config, Paths
from .logging import get_logger


def common_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--scene", default=None, help="scene name (defaults to $SCENE or config)")
    p.add_argument("--config", default="configs/default.yaml", help="path to pipeline config")
    return p


def setup(args) -> tuple[dict, Paths, "logging.Logger"]:
    cfg = load_config(args.config)
    paths = get_paths(cfg, args.scene).ensure()
    log = get_logger()
    log.info("scene=%s  root=%s", paths.scene, paths.root)
    return cfg, paths, log
