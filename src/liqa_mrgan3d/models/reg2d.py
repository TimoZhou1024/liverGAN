"""2D registration sub-network.

Mirrors src/liqa_mrgan3d/models/reg3d.py with 2D ops. Outputs a 2-channel
displacement field (one per spatial axis) that is consumed by Transformer2D.
The final layer is near-zero initialised so the network starts as identity.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _conv2d(
    in_c: int, out_c: int, *, kernel_size: int = 3, stride: int = 1, padding: int = 1
) -> nn.Conv2d:
    return nn.Conv2d(
        in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=True
    )


class ConvAct2D(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = _conv2d(in_c, out_c)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class ResnetBlock2D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _conv2d(dim, dim),
            nn.ReLU(inplace=True),
            _conv2d(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DownStage2D(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.block = ConvAct2D(in_c, out_c)
        self.res = ResnetBlock2D(out_c)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.res(self.block(x))
        skip = x
        return self.pool(x), skip


class UpStage2D(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = ConvAct2D(in_c + skip_c, out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class Reg2D(nn.Module):
    """2D dense displacement field predictor.

    ``forward(img_a, img_b)`` returns a flow ``[B, 2, H, W]`` describing how to
    deform ``img_a`` so it matches ``img_b``. Output layer is near-zero
    initialised so the network starts as the identity transform.
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
            self.downs.append(DownStage2D(in_c, out_c))
            in_c = out_c

        bottleneck = []
        for _ in range(n_resnet_blocks):
            bottleneck.append(ResnetBlock2D(in_c))
        self.bottleneck = nn.Sequential(*bottleneck)

        self.ups = nn.ModuleList()
        reversed_ndf = list(reversed(ndf))
        cur = in_c
        for skip_c, out_c in zip(reversed_ndf, reversed_ndf):
            self.ups.append(UpStage2D(cur, skip_c, out_c))
            cur = out_c

        self.refine = ConvAct2D(cur, cur)
        self.output = nn.Conv2d(cur, 2, kernel_size=3, padding=1, bias=True)
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
