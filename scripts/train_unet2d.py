"""Pretrain UNet2D on (GED4 slice, mask slice) pairs for the shape loss.

Mirrors scripts/train_unet3d.py but operates on 2D liver-bearing slices.
Output checkpoint goes to ``chk/UNet2D.pth`` (or whatever ``chk_path`` is in
the config) and is referenced as ``unet_chk`` in configs/liqa_2d.yaml.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from liqa_mrgan3d.data.datasets_liqa import LiQA2DSliceDataset
from liqa_mrgan3d.models.unet2d import UNet2D
from liqa_mrgan3d.trainer.losses import SoftDice3D, mask_to_onehot_2d
from liqa_mrgan3d.utils.config import ensure_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain UNet2D for the 2D shape loss.")
    parser.add_argument("--config", default="configs/unet2d.yaml")
    args = parser.parse_args()

    config: dict[str, Any] = load_config(args.config)
    device = torch.device(
        "cuda" if config.get("cuda", True) and torch.cuda.is_available() else "cpu"
    )
    output_dir = ensure_dir(config.get("save_root", "outputs/unet2d"))
    log_dir = ensure_dir(Path(output_dir) / "logs")
    chk_path = Path(config.get("chk_path", "chk/UNet2D.pth"))
    chk_path.parent.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        LiQA2DSliceDataset(config, split="train"),
        batch_size=int(config.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(config.get("n_cpu", 0)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    num_classes = int(config.get("num_classes", 2))
    palette = config.get("palette", [[0], [1]])

    unet = UNet2D(
        img_ch=1,
        num_classes=num_classes,
        depth=int(config.get("unet_depth", 1)),
        base_channels=int(config.get("unet_base_channels", 32)),
    ).to(device)

    dice_loss = SoftDice3D(num_classes).to(device)  # dim-agnostic
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=float(config.get("lr", 1e-4)))

    writer = SummaryWriter(log_dir=str(log_dir))
    step = 0
    for epoch in range(int(config.get("n_epochs", 30))):
        unet.train()
        progress = tqdm(train_loader, desc=f"unet2d epoch {epoch + 1}")
        for batch in progress:
            real_b = batch["B"].to(device, non_blocking=True)  # [B, 1, H, W] in [-1, 1]
            mask = batch["M"].to(device, non_blocking=True)  # [B, 1, H, W] binary

            logits = unet(real_b)
            probs = torch.sigmoid(logits)
            target_onehot = mask_to_onehot_2d(mask.long(), palette).to(device)

            loss_dice = dice_loss(probs, target_onehot)
            loss_bce = bce_loss(logits, target_onehot)
            loss = loss_dice + loss_bce

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/dice", loss_dice.item(), step)
            writer.add_scalar("train/bce", loss_bce.item(), step)
            progress.set_postfix(loss=f"{loss.item():.4f}")
            step += 1

        torch.save(unet.state_dict(), chk_path)
        print(f"saved unet2d weights to {chk_path}")

    writer.close()


if __name__ == "__main__":
    main()
