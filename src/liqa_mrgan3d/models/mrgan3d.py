"""3D port of reference/MrGAN/Model/Pix2Pix_v6.py.

Mirrors the Encoder / ShareNet / Decoder / Generator / Discriminator structure but
with Conv3d, InstanceNorm3d, AvgPool3d, and trilinear upsampling. The default
constructor matches the reference call site ``Generator(in_c, 64, 2, 3, True, True)``.
"""
from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
from torch import nn

from liqa_mrgan3d.models.cbam3d import CBAMBlock3D

BIAS = False


class ConvBlock3D(nn.Module):
    """Two-conv block; optionally returns half-channels skip + pooled output, or upsampled output."""

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
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=BIAS),
            nn.InstanceNorm3d(ch_out, affine=affine),
            actv,
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=BIAS),
            nn.InstanceNorm3d(ch_out, affine=affine),
            actv,
        )
        self.downsample = downsample
        self.upsample = upsample
        if self.upsample:
            self.up = UpConv3D(ch_out, ch_out // 2, affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x1 = self.conv(x)
        if self.downsample:
            x2 = F.avg_pool3d(x1, 2)
            c = x1.shape[1]
            return x1[:, : c // 2], x2
        if self.upsample:
            return self.up(x1)
        return x1


class UpConv3D(nn.Module):
    def __init__(
        self, ch_in: int, ch_out: int, affine: bool = True, actv: nn.Module | None = None
    ) -> None:
        super().__init__()
        if actv is None:
            actv = nn.LeakyReLU(inplace=True)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=BIAS),
            nn.InstanceNorm3d(ch_out, affine=affine),
            actv,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class Encoder3D(nn.Module):
    def __init__(self, input_nc: int, output_nc: int, layers: int, affine: bool) -> None:
        super().__init__()
        modules = []
        in_c, out_c = input_nc, output_nc
        for _ in range(layers):
            modules.append(ConvBlock3D(in_c, out_c, affine, downsample=True, upsample=False))
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


class ShareNet3D(nn.Module):
    def __init__(self, in_c: int, out_c: int, layers: int, affine: bool, r: int) -> None:
        super().__init__()
        encoders, decoders = [], []
        cur_in, cur_out, cur_r = in_c, out_c, r
        for _ in range(layers - 1):
            encoders.append(ConvBlock3D(cur_in, cur_in * 2, affine, downsample=True, upsample=False))
            decoders.append(ConvBlock3D(cur_out - cur_r, cur_out // 2, affine, downsample=False, upsample=True))
            cur_in = cur_in * 2
            cur_out = cur_out // 2
            cur_r = cur_r // 2
        self.bottom = ConvBlock3D(cur_in, cur_in * 2, affine, upsample=True)
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


class Decoder3D(nn.Module):
    def __init__(self, in_c: int, mid_c: int, layers: int, affine: bool, r: int) -> None:
        super().__init__()
        modules = []
        cur_in, cur_mid, cur_r = in_c, mid_c, r
        for _ in range(layers - 1):
            modules.append(
                nn.Sequential(
                    CBAMBlock3D(cur_in - cur_r),
                    ConvBlock3D(cur_in - cur_r, cur_mid, affine, downsample=False, upsample=True),
                )
            )
            cur_in = cur_mid
            cur_mid = cur_mid // 2
            cur_r = cur_r // 2
        self.conv_end = ConvBlock3D(cur_in - cur_r, cur_mid, affine, downsample=False, upsample=False)
        self.decoder = nn.ModuleList(modules)

    def forward(self, share_input: torch.Tensor, encoder_input: list[list[torch.Tensor]]) -> torch.Tensor:
        local_enc = list(encoder_input)
        local_enc.reverse()
        ii = len(self.decoder)
        for i, layer in enumerate(self.decoder):
            x = torch.cat([share_input, local_enc[i][0]], dim=1)
            x = layer(x)
            share_input = x
        x = torch.cat([share_input, local_enc[ii][0]], dim=1)
        return self.conv_end(x)


class MrGANGenerator3D(nn.Module):
    """3D Generator mirroring reference/MrGAN/Model/Pix2Pix_v6.py:143 ``Generator``."""

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
        self.img_encoder = Encoder3D(in_c, mid_c, layers, affine)
        # ``r`` mirrors the hard-coded values in the reference call site (64 and
        # 256), but parameterised by ``mid_c``/``layers`` so the channel arithmetic
        # remains consistent when ``mid_c`` is reduced for memory.
        decoder_r = mid_c
        share_r = mid_c * (2 ** layers)
        self.img_decoder = Decoder3D(
            mid_c * (2 ** layers),
            mid_c * (2 ** (layers - 1)),
            layers,
            affine,
            r=decoder_r,
        )
        self.share_net = ShareNet3D(
            mid_c * (2 ** (layers - 1)),
            mid_c * (2 ** (layers - 1 + s_layers)),
            s_layers,
            affine,
            r=share_r,
        )
        self.out_img = nn.Conv3d(mid_c, 1, kernel_size=1, bias=BIAS)
        self.last_ac = last_ac

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        enc = self.img_encoder(img)
        share = self.share_net(enc)
        out = self.out_img(self.img_decoder(share, enc))
        if self.last_ac:
            out = torch.tanh(out)
        return out


class MrGANDiscriminator3D(nn.Module):
    """3D PatchGAN discriminator (structurally mirrors reference/MrGAN/Model/Pix2Pix_v6.py:189).

    Patches in this project are thin along the depth axis (e.g. ``D=16``) so
    repeated stride-2 downsampling along D would collapse the volume before the
    final layer. We therefore use anisotropic kernels and strides: kernel
    ``(3, 4, 4)`` with stride ``(1, 2, 2)`` for the down-sampling layers and
    stride ``(1, 1, 1)`` for the final two PatchGAN convolutions. Spatial
    receptive field along H/W still matches the reference 70x70 PatchGAN.
    """

    def __init__(
        self,
        input_nc: int = 2,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: type[nn.Module] = nn.BatchNorm3d,
    ) -> None:
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = (3, 4, 4)
        padw = (1, 1, 1)
        ds_stride = (1, 2, 2)

        sequence: list[nn.Module] = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=ds_stride, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=ds_stride,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=(1, 1, 1),
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=(1, 1, 1), padding=padw),
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def mrgan_weights_init_normal(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, (nn.BatchNorm3d, nn.InstanceNorm3d)):
        if module.weight is not None:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
