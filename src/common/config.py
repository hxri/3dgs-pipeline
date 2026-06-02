"""Config + path resolution shared by every stage.

A stage is invoked as `python -m src.<stage> --scene <name> --config <yaml>`.
`load` returns the parsed config dict and a `Paths` object rooted at
`data/<scene>/` with one sub-directory per stage.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml


class Paths:
    """All on-disk locations for a single scene, derived from data_root + scene."""

    def __init__(self, data_root: str | Path, scene: str):
        self.scene = scene
        self.root = Path(data_root) / scene

    # raw input (video or image folder) lives under the scene root, e.g. raw/*.mp4
    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def frames(self) -> Path:
        return self.root / "frames"

    @property
    def colmap(self) -> Path:
        return self.root / "colmap"

    @property
    def splat(self) -> Path:
        return self.root / "splat"

    @property
    def labels(self) -> Path:
        return self.root / "labels"

    @property
    def render(self) -> Path:
        return self.root / "render"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def eval(self) -> Path:
        return self.root / "eval"

    def ensure(self) -> "Paths":
        for p in (self.frames, self.colmap, self.splat, self.labels,
                  self.render, self.models, self.eval):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def __repr__(self) -> str:
        return f"Paths(scene={self.scene!r}, root={self.root})"


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_scene(cfg: dict, scene: str | None) -> str:
    # precedence: CLI arg > SCENE env var > config default
    return scene or os.environ.get("SCENE") or cfg["project"]["scene"]


def get_paths(cfg: dict, scene: str | None = None) -> Paths:
    scene = resolve_scene(cfg, scene)
    return Paths(cfg["paths"]["data_root"], scene)
