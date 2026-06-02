"""Stage 03b — lift 2D semantic masks into 3D, then split classes into instances.

1. VOTE: project every Gaussian center into every seeded view and tally which class
   covers it. Each Gaussian's class = argmax votes (above a threshold, beating bg).
2. INSTANCE: within each class, spatially cluster Gaussians (DBSCAN) so two chairs in
   different parts of the room become two instances — no 2D tracking needed.

Output contract (consumed by stage 04 render):
    gaussian_instance_ids.npy   int[num_gaussians]   0 = background
    instance_classes.json       {instance_id: class_index}
"""
from __future__ import annotations

import numpy as np

from src.common.nerfstudio_io import project_points


def vote_classes(means, seeds, num_classes, device, log=None):
    """seeds: list of (label_map_tensor HxW int, fx, fy, cx, cy, W, H, c2w_tensor).
    Returns per-Gaussian class id tensor (0 = bg, 1..C) before thresholding plus the
    raw vote tensor (N, C+1)."""
    import torch

    n = means.shape[0]
    votes = torch.zeros((n, num_classes + 1), device=device)
    for i, (label_map, fx, fy, cx, cy, W, H, c2w) in enumerate(seeds):
        u, v, depth, valid = project_points(means, c2w, fx, fy, cx, cy, W, H)
        idx = valid.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        ui = u[idx].long()
        vi = v[idx].long()
        lbl = label_map[vi, ui]  # 0..C
        votes.index_put_((idx, lbl), torch.ones(idx.numel(), device=device), accumulate=True)
        if log and (i + 1) % 25 == 0:
            log.info("    voted %d/%d frames", i + 1, len(seeds))
    return votes


def assign_classes(votes, min_votes):
    """votes (N, C+1) -> per-Gaussian class id (0 bg, 1..C)."""
    import torch

    bg = votes[:, 0]
    fg = votes[:, 1:]
    best = fg.argmax(dim=1)            # 0..C-1
    cnt = fg.max(dim=1).values
    keep = (cnt >= min_votes) & (cnt > bg)
    return torch.where(keep, best + 1, torch.zeros_like(best))  # 1..C, else 0


def _cluster_one_class(pts: np.ndarray, eps: float, min_samples: int, max_pts: int):
    """Cluster points of a single class. Returns int[len(pts)] cluster index (-1 = noise)."""
    from sklearn.cluster import DBSCAN
    from scipy.spatial import cKDTree

    n = len(pts)
    fit_idx = (np.random.choice(n, max_pts, replace=False) if n > max_pts
               else np.arange(n))
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts[fit_idx])
    clusters = [c for c in np.unique(labels) if c != -1]
    if not clusters:
        return np.full(n, -1, dtype=np.int64)
    centroids = np.stack([pts[fit_idx][labels == c].mean(0) for c in clusters])
    # assign ALL points (incl. the ones we didn't fit on) to nearest centroid within 2*eps
    d, nn = cKDTree(centroids).query(pts)
    return np.where(d <= 2.0 * eps, nn, -1)


def cluster_instances(means_np, class_ids_np, num_classes, eps, min_samples, max_pts, log=None):
    """Return (instance_ids int[N], {instance_id: class_index})."""
    inst = np.zeros(len(class_ids_np), dtype=np.int32)
    inst_classes: dict[int, int] = {}
    next_id = 1
    for c in range(1, num_classes + 1):
        sel = np.where(class_ids_np == c)[0]
        if len(sel) == 0:
            continue
        local = _cluster_one_class(means_np[sel], eps, min_samples, max_pts)
        for cl in np.unique(local):
            if cl == -1:
                continue
            inst[sel[local == cl]] = next_id
            inst_classes[next_id] = c - 1          # store 0-based class index
            next_id += 1
        if log:
            log.info("    class %d -> %d instances", c, next_id - 1 - sum(
                1 for v in inst_classes.values() if v < c - 1))
    return inst, inst_classes
