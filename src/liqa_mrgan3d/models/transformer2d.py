"""2D spatial transformer.

Port of src/liqa_mrgan3d/models/transformer3d.py. Applies a 2-channel
displacement field via 4D ``F.grid_sample``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Transformer2D(nn.Module):
    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        # src, flow: [B, C, H, W]
        b = flow.shape[0]
        h, w = flow.shape[2], flow.shape[3]
        device = flow.device

        # Build identity index grid in (y, x) order matching flow channel layout.
        vectors = [torch.arange(0, s, device=device, dtype=torch.float32) for s in (h, w)]
        grids = torch.meshgrid(vectors, indexing="ij")  # tuple of (H, W)
        grid = torch.stack(grids, dim=0)  # [2, H, W]
        grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)  # [B, 2, H, W]
        new_locs = grid + flow

        # Normalize to [-1, 1] per axis.
        for i, size in enumerate((h, w)):
            new_locs[:, i] = 2.0 * (new_locs[:, i] / (size - 1) - 0.5)

        # F.grid_sample on 4D input expects (B, H, W, 2) with axis order (x, y).
        # Our layout is (y, x) so we permute and reverse.
        new_locs = new_locs.permute(0, 2, 3, 1)  # [B, H, W, 2] in (y, x)
        new_locs = new_locs[..., [1, 0]]  # → (x, y)
        return F.grid_sample(
            src, new_locs, align_corners=True, padding_mode="border", mode="bilinear"
        )
