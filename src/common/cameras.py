"""Novel-view camera trajectory generation for the render stage.

These produce camera-to-world 4x4 matrices (OpenGL/Nerfstudio convention:
+x right, +y up, -z forward) looking at a target point. The render stage feeds
these poses to the trained 3DGS model to synthesize labeled views.

Pure numpy — no GPU deps — so it's testable on its own:
    python -m src.common.cameras --preview
"""
from __future__ import annotations

import numpy as np


def look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Camera-to-world matrix for a camera at `eye` looking at `target`."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward /= (np.linalg.norm(forward) + 1e-9)
    right = np.cross(forward, up)
    right /= (np.linalg.norm(right) + 1e-9)
    true_up = np.cross(right, forward)

    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = -forward  # OpenGL: camera looks down -z
    c2w[:3, 3] = eye
    return c2w


def orbit(center, radius, n, elevation_deg=(-10, 35), seed=0) -> np.ndarray:
    """`n` cameras on a ring around `center`, azimuth swept 0..360, elevation sampled."""
    rng = np.random.default_rng(seed)
    center = np.asarray(center, dtype=np.float64)
    az = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    el = np.deg2rad(rng.uniform(elevation_deg[0], elevation_deg[1], size=n))
    poses = np.empty((n, 4, 4))
    for i in range(n):
        x = radius * np.cos(el[i]) * np.cos(az[i])
        z = radius * np.cos(el[i]) * np.sin(az[i])
        y = radius * np.sin(el[i])
        poses[i] = look_at(center + np.array([x, y, z]), center)
    return poses


def spiral(center, radius, n, turns=3.0, elevation_deg=(-10, 35)) -> np.ndarray:
    """Smooth spiral — good for video-style flythroughs."""
    center = np.asarray(center, dtype=np.float64)
    t = np.linspace(0.0, 1.0, n)
    az = 2.0 * np.pi * turns * t
    el = np.deg2rad(np.interp(t, [0, 1], elevation_deg))
    r = radius * (0.7 + 0.3 * np.cos(2 * np.pi * t))  # gently pulse distance
    poses = np.empty((n, 4, 4))
    for i in range(n):
        x = r[i] * np.cos(el[i]) * np.cos(az[i])
        z = r[i] * np.cos(el[i]) * np.sin(az[i])
        y = r[i] * np.sin(el[i])
        poses[i] = look_at(center + np.array([x, y, z]), center)
    return poses


def hemisphere(center, radius, n, elevation_deg=(5, 80), seed=0) -> np.ndarray:
    """Random cameras over an upper hemisphere — broad coverage, less smooth."""
    rng = np.random.default_rng(seed)
    center = np.asarray(center, dtype=np.float64)
    az = rng.uniform(0, 2 * np.pi, n)
    el = np.deg2rad(rng.uniform(elevation_deg[0], elevation_deg[1], n))
    poses = np.empty((n, 4, 4))
    for i in range(n):
        x = radius * np.cos(el[i]) * np.cos(az[i])
        z = radius * np.cos(el[i]) * np.sin(az[i])
        y = radius * np.sin(el[i])
        poses[i] = look_at(center + np.array([x, y, z]), center)
    return poses


def jitter_input(input_c2w: np.ndarray, n, trans_sigma=0.05, rot_sigma_deg=3.0,
                 seed=0) -> np.ndarray:
    """Perturb real input poses slightly — stays inside the well-reconstructed region,
    so masks are most reliable. Safest trajectory for a first run."""
    rng = np.random.default_rng(seed)
    input_c2w = np.asarray(input_c2w, dtype=np.float64).reshape(-1, 4, 4)
    out = np.empty((n, 4, 4))
    for i in range(n):
        base = input_c2w[rng.integers(len(input_c2w))].copy()
        base[:3, 3] += rng.normal(0, trans_sigma, 3)
        ang = np.deg2rad(rng.normal(0, rot_sigma_deg, 3))
        out[i] = base @ _rot(ang)
    return out


def make_trajectory(name: str, center, radius, n, elevation_deg=(-10, 35),
                    input_c2w=None, seed=0) -> np.ndarray:
    name = name.lower()
    if name == "orbit":
        return orbit(center, radius, n, elevation_deg, seed)
    if name == "spiral":
        return spiral(center, radius, n, elevation_deg=elevation_deg)
    if name == "hemisphere":
        return hemisphere(center, radius, n, elevation_deg, seed)
    if name == "jitter_input":
        if input_c2w is None:
            raise ValueError("jitter_input trajectory needs input_c2w poses")
        return jitter_input(input_c2w, n, seed=seed)
    raise ValueError(f"unknown trajectory: {name!r}")


def _rot(ang) -> np.ndarray:
    rx, ry, rz = ang
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3] = Rz @ Ry @ Rx
    return m


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--traj", default="orbit")
    ap.add_argument("-n", type=int, default=8)
    a = ap.parse_args()
    P = make_trajectory(a.traj, center=[0, 0, 0], radius=3.0, n=a.n,
                        input_c2w=np.eye(4)[None] if a.traj == "jitter_input" else None)
    print(f"{a.traj}: {P.shape[0]} poses")
    for i, p in enumerate(P):
        print(f"  cam{i}: eye={np.round(p[:3, 3], 3)}")
