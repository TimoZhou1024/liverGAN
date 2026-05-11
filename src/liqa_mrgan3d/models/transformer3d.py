"""3D spatial transformer.

Mirrors reference/MrGAN/trainer/transformer.py:Transformer_2D, generalised to 3D.
``flow`` is a per-voxel displacement field with 3 channels (one per axis).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Transformer3D(nn.Module):
    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        # src, flow: [B, C, D, H, W]
        b = flow.shape[0]
        d, h, w = flow.shape[2], flow.shape[3], flow.shape[4]
        device = flow.device

        # Build identity index grid in (z, y, x) order matching flow axis layout.
        vectors = [torch.arange(0, s, device=device, dtype=torch.float32) for s in (d, h, w)]
        grids = torch.meshgrid(vectors, indexing="ij")  # tuple of (D, H, W)
        grid = torch.stack(grids, dim=0)  # [3, D, H, W]
        grid = grid.unsqueeze(0).repeat(b, 1, 1, 1, 1)  # [B, 3, D, H, W]
        new_locs = grid + flow

        # Normalize to [-1, 1] per axis.
        shape = (d, h, w)
        for i in range(3):
            new_locs[:, i] = 2.0 * (new_locs[:, i] / (shape[i] - 1) - 0.5)

        # F.grid_sample for 5D input expects (B, D, H, W, 3) with axis order (x, y, z).
        # Our layout is (z, y, x) so we permute and reverse channels.
        new_locs = new_locs.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3] in (z, y, x)
        new_locs = new_locs[..., [2, 1, 0]]  # → (x, y, z)
        return F.grid_sample(src, new_locs, align_corners=True, padding_mode="border", mode="bilinear")
