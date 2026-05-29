"""2D port of MrGAN Generator + Discriminator.

Mirrors the structure of src/liqa_mrgan3d/models/mrgan3d.py, swapping every 3D
op for its 2D counterpart. Algorithm is identical to the original 2D MrGAN
(reference/MrGAN/Model/Pix2Pix_v6.py): Encoder → ShareNet → CBAM Decoder.
"""
from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
from torch import nn

from liqa_mrgan3d.models.cbam2d import CBAMBlock2D

BIAS = False


class ConvBlock2D(nn.Module):
    """Two-conv block; optionally returns half-channels skip + pooled output."""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        affine: bool = True,
        actv: nn.Module | None = None,
        downsample: bool = False,
        upsample: bool = False,
    ) -> None:
        super().__init__()
        if actv is None:
            actv = nn.LeakyReLU(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=BIAS),
            nn.InstanceNorm2d(ch_out, affine=affine),
            actv,
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=BIAS),
            nn.InstanceNorm2d(ch_out, affine=affine),
            actv,
        )
        self.downsample = downsample
        self.upsample = upsample
        if self.upsample:
            self.up = UpConv2D(ch_out, ch_out // 2, affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x1 = self.conv(x)
        if self.downsample:
            x2 = F.avg_pool2d(x1, 2)
            c = x1.shape[1]
            return x1[:, : c // 2], x2
        if self.upsample:
            return self.up(x1)
        return x1


class UpConv2D(nn.Module):
    def __init__(
        self, ch_in: int, ch_out: int, affine: bool = True, actv: nn.Module | None = None
    ) -> None:
        super().__init__()
        if actv is None:
            actv = nn.LeakyReLU(inplace=True)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=BIAS),
            nn.InstanceNorm2d(ch_out, affine=affine),
            actv,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class Encoder2D(nn.Module):
    def __init__(self, input_nc: int, output_nc: int, layers: int, affine: bool) -> None:
        super().__init__()
        modules = []
        in_c, out_c = input_nc, output_nc
        for _ in range(layers):
            modules.append(ConvBlock2D(in_c, out_c, affine, downsample=True, upsample=False))
            in_c = out_c
            out_c = out_c * 2
        self.encoder = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> list[list[torch.Tensor]]:
        res: list[list[torch.Tensor]] = []
        for layer in self.encoder:
            x1, x2 = layer(x)
            res.append([x1, x2])
            x = x2
        return res


class ShareNet2D(nn.Module):
    def __init__(self, in_c: int, out_c: int, layers: int, affine: bool, r: int) -> None:
        super().__init__()
        encoders, decoders = [], []
        cur_in, cur_out, cur_r = in_c, out_c, r
        for _ in range(layers - 1):
            encoders.append(ConvBlock2D(cur_in, cur_in * 2, affine, downsample=True, upsample=False))
            decoders.append(
                ConvBlock2D(cur_out - cur_r, cur_out // 2, affine, downsample=False, upsample=True)
            )
            cur_in = cur_in * 2
            cur_out = cur_out // 2
            cur_r = cur_r // 2
        self.bottom = ConvBlock2D(cur_in, cur_in * 2, affine, upsample=True)
        self.encoder = nn.ModuleList(encoders)
        self.decoder = nn.ModuleList(decoders)
        self.layers = layers

    def forward(self, encoder_out: list[list[torch.Tensor]]) -> torch.Tensor:
        local_encoder_out: list[list[torch.Tensor]] = []
        x = encoder_out[-1][1]
        for layer in self.encoder:
            x1, x2 = layer(x)
            local_encoder_out.append([x1, x2])
            x = x2
        bottom = self.bottom(x)
        if self.layers == 1:
            return bottom
        local_encoder_out.reverse()
        for i, layer in enumerate(self.decoder):
            x = torch.cat([bottom, local_encoder_out[i][0]], dim=1)
            x = layer(x)
            bottom = x
        return bottom


class Decoder2D(nn.Module):
    def __init__(self, in_c: int, mid_c: int, layers: int, affine: bool, r: int) -> None:
        super().__init__()
        modules = []
        cur_in, cur_mid, cur_r = in_c, mid_c, r
        for _ in range(layers - 1):
            modules.append(
                nn.Sequential(
                    CBAMBlock2D(cur_in - cur_r),
                    ConvBlock2D(cur_in - cur_r, cur_mid, affine, downsample=False, upsample=True),
                )
            )
            cur_in = cur_mid
            cur_mid = cur_mid // 2
            cur_r = cur_r // 2
        self.conv_end = ConvBlock2D(cur_in - cur_r, cur_mid, affine, downsample=False, upsample=False)
        self.decoder = nn.ModuleList(modules)

    def forward(
        self, share_input: torch.Tensor, encoder_input: list[list[torch.Tensor]]
    ) -> torch.Tensor:
        local_enc = list(encoder_input)
        local_enc.reverse()
        ii = len(self.decoder)
        for i, layer in enumerate(self.decoder):
            x = torch.cat([share_input, local_enc[i][0]], dim=1)
            x = layer(x)
            share_input = x
        x = torch.cat([share_input, local_enc[ii][0]], dim=1)
        return self.conv_end(x)


class MrGANGenerator2D(nn.Module):
    """2D Generator mirroring reference/MrGAN/Model/Pix2Pix_v6.py:Generator."""

    def __init__(
        self,
        in_c: int,
        mid_c: int = 64,
        layers: int = 2,
        s_layers: int = 3,
        affine: bool = True,
        last_ac: bool = True,
    ) -> None:
        super().__init__()
        self.img_encoder = Encoder2D(in_c, mid_c, layers, affine)
        decoder_r = mid_c
        share_r = mid_c * (2 ** layers)
        self.img_decoder = Decoder2D(
            mid_c * (2 ** layers),
            mid_c * (2 ** (layers - 1)),
            layers,
            affine,
            r=decoder_r,
        )
        self.share_net = ShareNet2D(
            mid_c * (2 ** (layers - 1)),
            mid_c * (2 ** (layers - 1 + s_layers)),
            s_layers,
            affine,
            r=share_r,
        )
        self.out_img = nn.Conv2d(mid_c, 1, kernel_size=1, bias=BIAS)
        self.last_ac = last_ac

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        enc = self.img_encoder(img)
        share = self.share_net(enc)
        out = self.out_img(self.img_decoder(share, enc))
        if self.last_ac:
            out = torch.tanh(out)
        return out


class MrGANDiscriminator2D(nn.Module):
    """2D PatchGAN discriminator (mirrors reference/MrGAN/Model/Pix2Pix_v6.py:Discriminator).

    Standard isotropic stride-2 with kernel 4 — the 2D version doesn't have the
    thin-D-axis issue that the 3D port had to work around.
    """

    def __init__(
        self,
        input_nc: int = 2,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw, padw = 4, 1
        sequence: list[nn.Module] = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
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
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def mrgan_weights_init_normal_2d(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        if module.weight is not None:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
