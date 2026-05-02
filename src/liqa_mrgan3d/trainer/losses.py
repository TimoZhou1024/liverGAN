from __future__ import annotations

import math

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


def psnr_from_mse(mse: float, data_range: float = 2.0) -> float:
    if mse <= 0:
        return math.inf
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)
