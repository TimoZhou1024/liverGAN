from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, affine: bool = True) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=affine),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=affine),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock3D(in_channels, out_channels)
        self.pool = nn.AvgPool3d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return skip, self.pool(skip)


class UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock3D(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class Generator3D(nn.Module):
    """Patch-based 3D U-Net generator.

    Input: [B, C, D, H, W]
    Output: [B, 1, D, H, W]
    """

    def __init__(self, input_nc: int, output_nc: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        self.down1 = DownBlock3D(input_nc, c)
        self.down2 = DownBlock3D(c, c * 2)
        self.down3 = DownBlock3D(c * 2, c * 4)
        self.center = ConvBlock3D(c * 4, c * 8)
        self.up3 = UpBlock3D(c * 8, c * 4, c * 4)
        self.up2 = UpBlock3D(c * 4, c * 2, c * 2)
        self.up1 = UpBlock3D(c * 2, c, c)
        self.out = nn.Conv3d(c, output_nc, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, x = self.down1(x)
        s2, x = self.down2(x)
        s3, x = self.down3(x)
        x = self.center(x)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return torch.tanh(self.out(x))


class PatchDiscriminator3D(nn.Module):
    """3D PatchGAN discriminator."""

    def __init__(
        self,
        input_nc: int,
        ndf: int = 16,
        n_layers: int = 3,
        norm_layer: type[nn.Module] = nn.BatchNorm3d,
    ) -> None:
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        sequence: list[nn.Module] = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        final_kw = 3
        final_pad = 1
        sequence += [
            nn.Conv3d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=final_kw,
                stride=1,
                padding=final_pad,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=final_kw, stride=1, padding=final_pad),
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def weights_init_normal_3d(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm3d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)
