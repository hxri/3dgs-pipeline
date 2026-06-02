"""Stage 04 core — instance-aware 3DGS rasterization via gsplat.

Two rasterization passes per view, sharing identical geometry so they align pixel-perfect:
  1. RGB    — SH colors, alpha-composited (matches splatfacto appearance).
  2. INSTANCE — a one-hot-per-instance feature buffer. Per pixel we take argmax over the
     instance channels (above a coverage threshold) to get a crisp instance id rather than
     a meaningless blended id.

GPU + gsplat required.
"""
from __future__ import annotations

import numpy as np

from src.common.nerfstudio_io import (activated_gaussians, get_viewmat,
                                       scene_bounds, sh_colors_and_degree)


class RenderEngine:
    def __init__(self, model, instance_ids: np.ndarray, device):
        import torch

        self.device = device
        self.means, self.quats, self.scales, self.opac = activated_gaussians(model)
        self.sh_colors, self.sh_degree = sh_colors_and_degree(model)

        inst = torch.as_tensor(instance_ids, device=device, dtype=torch.long)
        self.num_instances = int(inst.max().item())
        # one-hot feature buffer: channel j == instance id (j+1); background -> all zeros
        self.onehot = torch.zeros((self.means.shape[0], max(1, self.num_instances)),
                                  device=device, dtype=torch.float32)
        fg = inst > 0
        self.onehot[fg, inst[fg] - 1] = 1.0

        self.scene_center, self.scene_radius = scene_bounds(self.means)

    def _Ks(self, fx, fy, cx, cy):
        import torch
        return torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]],
                            device=self.device, dtype=torch.float32)

    def render(self, c2w: np.ndarray, W: int, H: int, fx, fy, cx, cy,
               alpha_thresh: float = 0.5):
        """Return (rgb uint8 HxWx3, instance_map int HxW; 0 = background)."""
        import torch
        from gsplat.rendering import rasterization

        c2w_t = torch.as_tensor(c2w, dtype=torch.float32, device=self.device)
        viewmat = get_viewmat(c2w_t)[None]            # (1,4,4)
        Ks = self._Ks(fx, fy, cx, cy)                 # (1,3,3)

        # 1) RGB (SH)
        rgb, _, _ = rasterization(
            self.means, self.quats, self.scales, self.opac, self.sh_colors,
            viewmat, Ks, W, H, sh_degree=self.sh_degree, render_mode="RGB")
        rgb = torch.clamp(rgb[0], 0.0, 1.0)

        # 2) instance one-hot
        inst_col, inst_alpha, _ = rasterization(
            self.means, self.quats, self.scales, self.opac, self.onehot,
            viewmat, Ks, W, H, sh_degree=None, render_mode="RGB")
        coverage, arg = inst_col[0].max(dim=-1)       # (H,W)
        alpha = inst_alpha[0, ..., 0]                 # (H,W)
        fg = (coverage > alpha_thresh) & (alpha > alpha_thresh)
        inst_map = torch.where(fg, arg + 1, torch.zeros_like(arg))

        rgb_np = (rgb * 255.0).round().to(torch.uint8).cpu().numpy()
        return rgb_np, inst_map.to(torch.int32).cpu().numpy()
