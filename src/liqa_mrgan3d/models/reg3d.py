"""3D registration sub-network.

Algorithmic counterpart of reference/MrGAN/trainer/reg.py. The original is a 2D
ResUnet producing a 2-channel flow field. We adapt this to 3D by:

* Replacing 2D ops with 3D equivalents.
* Reducing the depth of the down-sampling pyramid (original = 7 stages with
  pool-by-2) since a 16-slice patch cannot survive more than four halvings.
* Outputting a 3-channel displacement field (one per spatial axis).

The structure remains "Down -> ResNet bottleneck -> Up with skip" so the receptive
field and residual-block bottleneck character match the reference.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _conv3d(in_c: int, out_c: int, *, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)


class ConvAct3D(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = _conv3d(in_c, out_c)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class ResnetBlock3D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _conv3d(dim, dim),
            nn.ReLU(inplace=True),
            _conv3d(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DownStage3D(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.block = ConvAct3D(in_c, out_c)
        self.res = ResnetBlock3D(out_c)
        self.pool = nn.AvgPool3d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.res(self.block(x))
        skip = x
        return self.pool(x), skip


class UpStage3D(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = ConvAct3D(in_c + skip_c, out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class Reg3D(nn.Module):
    """3D dense displacement field predictor.

    ``forward(img_a, img_b)`` returns a flow ``[B, 3, D, H, W]`` describing how
    to deform ``img_a`` so it matches ``img_b``. Final layer is zero-initialised
    so the network starts as the identity transform (matches the
    ``init_to_identity`` flag in the reference).
    """

    def __init__(
        self,
        in_channels_a: int = 1,
        in_channels_b: int = 1,
        ndf: tuple[int, ...] = (32, 64, 64, 64),
        n_resnet_blocks: int = 3,
    ) -> None:
        super().__init__()
        in_c = in_channels_a + in_channels_b
        self.downs = nn.ModuleList()
        for out_c in ndf:
            self.downs.append(DownStage3D(in_c, out_c))
            in_c = out_c

        bottleneck = []
        for _ in range(n_resnet_blocks):
            bottleneck.append(ResnetBlock3D(in_c))
        self.bottleneck = nn.Sequential(*bottleneck)

        self.ups = nn.ModuleList()
        reversed_ndf = list(reversed(ndf))
        cur = in_c
        for skip_c, out_c in zip(reversed_ndf, reversed_ndf):
            self.ups.append(UpStage3D(cur, skip_c, out_c))
            cur = out_c

        self.refine = ConvAct3D(cur, cur)
        self.output = nn.Conv3d(cur, 3, kernel_size=3, padding=1, bias=True)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.output.bias)

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([img_a, img_b], dim=1)
        skips: list[torch.Tensor] = []
        for stage in self.downs:
            x, skip = stage(x)
            skips.append(skip)
        x = self.bottleneck(x)
        for stage, skip in zip(self.ups, reversed(skips)):
            x = stage(x, skip)
        x = self.refine(x)
        return self.output(x)
