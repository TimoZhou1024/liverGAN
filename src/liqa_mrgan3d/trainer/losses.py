from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class GANLoss(nn.Module):
    def __init__(self, target_real_label: float = 1.0, target_fake_label: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = self.real_label if target_is_real else self.fake_label
        return target.expand_as(prediction)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        return self.loss(prediction, self.get_target_tensor(prediction, target_is_real))


class GaussianBlur2D(nn.Module):
    def __init__(self, channels: int = 1, kernel_size: int = 5, sigma: float = 1.0) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        self.register_buffer("kernel", kernel)
        self.groups = channels
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(x, self.kernel, padding=self.padding, groups=self.groups)


class GaussianBlur3D(nn.Module):
    def __init__(self, channels: int = 1, kernel_size: int = 5, sigma: float = 1.0) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        grid_z, grid_y, grid_x = torch.meshgrid(coords, coords, coords, indexing="ij")
        kernel = torch.exp(-(grid_x**2 + grid_y**2 + grid_z**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size).repeat(
            channels, 1, 1, 1, 1
        )
        self.register_buffer("kernel", kernel)
        self.groups = channels
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv3d(x, self.kernel, padding=self.padding, groups=self.groups)


class SmoothnessLoss3D(nn.Module):
    """Squared first-order finite-difference penalty on a 3D displacement field.

    Mirrors ``smooothing_loss`` from reference/MrGAN/trainer/utils.py:167, generalised
    from (B, C, H, W) to (B, C, D, H, W).
    """

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        dz = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
        dy = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
        dx = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
        return (dz * dz).mean() + (dy * dy).mean() + (dx * dx).mean()


class LPIPS2DSliceWise(nn.Module):
    """LPIPS computed slice-wise along the depth axis of a 3D volume.

    LPIPS (lpips package, VGG backbone) is a 2D-only perceptual metric. To use
    it on 3D volumes we evaluate it on every axial slice ``[B, 1, H, W]``,
    replicate the single grayscale channel to 3 channels, clip to [-1, 1], and
    average across slices. Inputs are expected in tanh range (~[-1, 1]).

    At ``D=32`` a single call effectively runs a ``batch=32`` VGG16 forward per
    LPIPS invocation, which dominates activation memory on large volumes. Set
    ``num_slices`` to randomly subsample ``k`` slices per call (stochastic
    Monte Carlo estimator of the per-slice mean) — this cuts VGG memory by
    ``D / k``.
    """

    def __init__(self, net: str = "vgg", num_slices: int | None = None) -> None:
        super().__init__()
        import lpips  # local import - heavy

        self.model = lpips.LPIPS(net=net)
        for p in self.model.parameters():
            p.requires_grad = False
        self.num_slices = num_slices

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred / target: [B, 1, D, H, W]
        if pred.dim() != 5 or target.dim() != 5:
            raise ValueError(f"Expected 5D inputs, got {pred.shape} and {target.shape}")
        b, c, d, h, w = pred.shape

        if self.num_slices is not None and 0 < self.num_slices < d:
            # Random subset of slices; sort indices so downstream reshape is deterministic.
            idx = torch.randperm(d, device=pred.device)[: self.num_slices].sort().values
            pred = pred.index_select(2, idx)
            target = target.index_select(2, idx)
            d = self.num_slices

        pred_2d = pred.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        target_2d = target.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        if c == 1:
            pred_2d = pred_2d.repeat(1, 3, 1, 1)
            target_2d = target_2d.repeat(1, 3, 1, 1)
        pred_2d = pred_2d.clamp(-1.0, 1.0)
        target_2d = target_2d.clamp(-1.0, 1.0)
        distances = self.model.forward(pred_2d, target_2d)
        return distances.mean()


class SoftDice3D(nn.Module):
    """SoftDice loss over foreground classes for 3D one-hot inputs.

    Mirrors ``SoftDiceLoss`` from reference/MrGAN/trainer/utils.py:311. Skips
    background channel (index 0), averages Dice over remaining classes, returns
    ``1 - mean_dice``.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    @staticmethod
    def _dice_coef(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        n = target.size(0)
        pred_flat = pred.reshape(n, -1)
        target_flat = target.reshape(n, -1)
        tp = (pred_flat * target_flat).sum(dim=1)
        fp = pred_flat.sum(dim=1) - tp
        fn = target_flat.sum(dim=1) - tp
        score = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        return score.sum() / n

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        class_dice = []
        for i in range(1, self.num_classes):
            class_dice.append(self._dice_coef(y_pred[:, i : i + 1], y_true[:, i : i + 1]))
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice


def mask_to_onehot_3d(mask: torch.Tensor, palette: list[list[int]]) -> torch.Tensor:
    """Convert a class-id mask to a one-hot tensor, fully on-device.

    ``mask`` has shape ``[B, 1, D, H, W]`` (or ``[1, D, H, W]``) with integer
    class ids. ``palette`` is a list of single-element class ids, e.g.
    ``[[0], [1]]``. Returns a float tensor of shape
    ``[B, len(palette), D, H, W]`` on the same device as ``mask``.

    The older version copied to CPU via NumPy on every call; that CPU sync
    serialises with distributed all-reduce collectives and kills FSDP
    throughput, so the implementation is now a pure-torch broadcast comparison.
    """
    if mask.dim() == 4:
        mask = mask.unsqueeze(0)
    if mask.dim() != 5 or mask.shape[1] != 1:
        raise ValueError(f"mask must have shape [B, 1, D, H, W], got {tuple(mask.shape)}")
    channel = mask[:, 0]  # [B, D, H, W]
    class_ids = [colour[0] for colour in palette]
    return torch.stack(
        [(channel == c).to(torch.float32) for c in class_ids],
        dim=1,
    )


def psnr_from_mse(mse: float, data_range: float = 2.0) -> float:
    if mse <= 0:
        return math.inf
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)
