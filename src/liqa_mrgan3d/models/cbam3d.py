"""3D port of the CBAM attention module from reference/MrGAN/Model/CBAM.py."""
from __future__ import annotations

import torch
from torch import nn


class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 8) -> None:
        super().__init__()
        hidden = max(1, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc1 = nn.Conv3d(in_planes, hidden, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(hidden, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


class CBAMBlock3D(nn.Module):
    def __init__(self, channel: int, ratio: int = 8, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention3D(channel, ratio=ratio)
        self.spatial_attention = SpatialAttention3D(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x
