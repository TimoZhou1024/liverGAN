"""2D UNet for the MrGAN shape loss.

Mirrors src/liqa_mrgan3d/models/unet3d.py — same channel ladder, configurable
depth and base_channels. Used by ``LiQA2DTrainer`` for the SoftDice shape loss
on liver masks.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class EncoderBlock2D(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            )
        ]
        for _ in range(1, depth):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(inplace=True),
                )
            )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderBlock2D(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, use_deconv: bool = False) -> None:
        super().__init__()
        if use_deconv:
            self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class UNet2D(nn.Module):
    def __init__(
        self,
        img_ch: int = 1,
        num_classes: int = 2,
        depth: int = 1,
        use_deconv: bool = False,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        chs = [base_channels * m for m in (1, 2, 4, 8, 16)]
        self.pool = nn.MaxPool2d(2, 2)

        self.enc1 = EncoderBlock2D(img_ch, chs[0], depth=depth)
        self.enc2 = EncoderBlock2D(chs[0], chs[1], depth=depth)
        self.enc3 = EncoderBlock2D(chs[1], chs[2], depth=depth)
        self.enc4 = EncoderBlock2D(chs[2], chs[3], depth=depth)
        self.center = EncoderBlock2D(chs[3], chs[4], depth=depth)

        self.dec4 = DecoderBlock2D(chs[4], chs[3], use_deconv=use_deconv)
        self.decconv4 = EncoderBlock2D(chs[3] * 2, chs[3])
        self.dec3 = DecoderBlock2D(chs[3], chs[2], use_deconv=use_deconv)
        self.decconv3 = EncoderBlock2D(chs[2] * 2, chs[2])
        self.dec2 = DecoderBlock2D(chs[2], chs[1], use_deconv=use_deconv)
        self.decconv2 = EncoderBlock2D(chs[1] * 2, chs[1])
        self.dec1 = DecoderBlock2D(chs[1], chs[0], use_deconv=use_deconv)
        self.decconv1 = EncoderBlock2D(chs[0] * 2, chs[0])

        self.conv_1x1 = nn.Conv2d(chs[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        center = self.center(self.pool(x4))

        d4 = self.dec4(center)
        if d4.shape[2:] != x4.shape[2:]:
            d4 = F.interpolate(d4, size=x4.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.decconv4(torch.cat([x4, d4], dim=1))

        d3 = self.dec3(d4)
        if d3.shape[2:] != x3.shape[2:]:
            d3 = F.interpolate(d3, size=x3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.decconv3(torch.cat([x3, d3], dim=1))

        d2 = self.dec2(d3)
        if d2.shape[2:] != x2.shape[2:]:
            d2 = F.interpolate(d2, size=x2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.decconv2(torch.cat([x2, d2], dim=1))

        d1 = self.dec1(d2)
        if d1.shape[2:] != x1.shape[2:]:
            d1 = F.interpolate(d1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.decconv1(torch.cat([x1, d1], dim=1))

        return self.conv_1x1(d1)
