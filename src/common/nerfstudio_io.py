"""Shared glue for loading a trained nerfstudio splatfacto scene and working with its
Gaussians + cameras in a single consistent coordinate frame.

Both stage 03 (label) and stage 04 (render) need: the Gaussian parameters, the training
cameras (poses + intrinsics) in the *same* normalized frame as the Gaussians, and a few
projection/render conventions. Centralizing them here keeps those conventions identical
across stages (a common source of silent misalignment bugs).

GPU-only (needs torch + nerfstudio + gsplat). Imports are lazy so the module is
import-safe on a CPU box.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# ── loading ──────────────────────────────────────────────────────────────────
def load_pipeline(config_path: str | Path):
    """Load a trained nerfstudio pipeline in inference mode. Returns (config, pipeline)."""
    import torch  # noqa: F401
    from nerfstudio.utils.eval_utils import eval_setup

    result = eval_setup(Path(config_path), test_mode="inference")
    config, pipeline = result[0], result[1]
    pipeline.eval()
    return config, pipeline


def resolve_latest_config(splat_dir: Path) -> Path:
    """Stage 02 writes splat/LATEST_CONFIG pointing at the trained config.yml."""
    ptr = Path(splat_dir) / "LATEST_CONFIG"
    if ptr.exists():
        return Path(ptr.read_text().strip())
    candidates = sorted(Path(splat_dir).rglob("config.yml"))
    if not candidates:
        raise FileNotFoundError(f"No trained config under {splat_dir} (run stage 02)")
    return candidates[-1]


# ── gaussian parameters ──────────────────────────────────────────────────────
def gauss_params(model) -> dict:
    """splatfacto stores raw (pre-activation) params in model.gauss_params."""
    gp = getattr(model, "gauss_params", None)
    if gp is None:  # very old nerfstudio fallback
        gp = {k: getattr(model, k) for k in
              ("means", "scales", "quats", "features_dc", "features_rest", "opacities")}
    return gp


def activated_gaussians(model):
    """Return (means, quats_normalized, scales_exp, opacities_sigmoid) ready for gsplat."""
    import torch

    gp = gauss_params(model)
    means = gp["means"]
    quats = torch.nn.functional.normalize(gp["quats"], dim=-1)
    scales = torch.exp(gp["scales"])
    opac = torch.sigmoid(gp["opacities"]).squeeze(-1)
    return means, quats, scales, opac


def sh_colors_and_degree(model):
    """Spherical-harmonic colors (N, (deg+1)^2, 3) plus the SH degree, for RGB render."""
    import torch

    gp = gauss_params(model)
    colors = torch.cat([gp["features_dc"][:, None, :], gp["features_rest"]], dim=1)
    n_coeffs = colors.shape[1]
    sh_degree = int(round(n_coeffs ** 0.5)) - 1
    return colors, sh_degree


def scene_bounds(means, radius_quantile: float = 0.9):
    """Robust scene center + radius from Gaussian centers (ignores far floaters)."""
    import torch

    center = means.median(dim=0).values
    dist = torch.linalg.norm(means - center, dim=-1)
    radius = torch.quantile(dist, radius_quantile)
    return center.detach().cpu().numpy(), float(radius)


# ── cameras ──────────────────────────────────────────────────────────────────
def get_train_cameras(pipeline):
    """Return (cameras, image_paths, filenames) for the training views, in the same
    normalized frame as the Gaussians."""
    ds = pipeline.datamanager.train_dataset
    cams = ds.cameras
    outputs = getattr(ds, "_dataparser_outputs", None) or ds.dataparser_outputs
    image_paths = [str(p) for p in outputs.image_filenames]
    filenames = [Path(p).name for p in image_paths]
    return cams, image_paths, filenames


def camera_to_world_4x4(cams, idx: int) -> np.ndarray:
    """4x4 camera-to-world (OpenGL convention: +x right, +y up, -z forward)."""
    c2w = cams.camera_to_worlds[idx].detach().cpu().numpy()  # (3,4)
    out = np.eye(4)
    out[:3, :4] = c2w
    return out


def intrinsics(cams, idx: int):
    """(fx, fy, cx, cy, W, H) for one training camera."""
    return (float(cams.fx[idx]), float(cams.fy[idx]),
            float(cams.cx[idx]), float(cams.cy[idx]),
            int(cams.width[idx]), int(cams.height[idx]))


def intrinsics_scaled(cams, idx: int, target_wh):
    """Reference intrinsics rescaled to a target (W, H) render resolution."""
    fx, fy, cx, cy, W0, H0 = intrinsics(cams, idx)
    W, H = target_wh
    sx, sy = W / W0, H / H0
    return fx * sx, fy * sy, cx * sx, cy * sy, W, H


# ── render conventions ───────────────────────────────────────────────────────
def get_viewmat(c2w):
    """OpenGL camera-to-world (4x4) -> OpenCV world-to-camera viewmat for gsplat.
    Mirrors nerfstudio splatfacto's get_viewmat (flip y,z camera axes)."""
    import torch

    if not torch.is_tensor(c2w):
        c2w = torch.tensor(c2w, dtype=torch.float32)
    R = c2w[:3, :3].clone()
    T = c2w[:3, 3:4].clone()
    R = R * torch.tensor([1.0, -1.0, -1.0], device=R.device, dtype=R.dtype)  # GL->CV
    R_inv = R.transpose(0, 1)
    T_inv = -R_inv @ T
    viewmat = torch.eye(4, device=c2w.device, dtype=c2w.dtype)
    viewmat[:3, :3] = R_inv
    viewmat[:3, 3:4] = T_inv
    return viewmat


def project_points(means, c2w, fx, fy, cx, cy, W, H):
    """Project world points into one OpenGL camera. Torch in/out.
    Returns (u, v, depth, valid) where valid is in-front-and-in-frame."""
    import torch

    R = c2w[:3, :3]
    t = c2w[:3, 3]
    pc = (means - t) @ R                 # R^T (p - t)
    depth = -pc[:, 2]                    # OpenGL looks down -z; depth>0 in front
    u = fx * (pc[:, 0] / depth) + cx
    v = fy * (-pc[:, 1] / depth) + cy    # image v grows downward
    valid = (depth > 1e-4) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, depth, valid
