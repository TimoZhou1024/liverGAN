from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from liqa_mrgan3d.data.datasets_liqa import LiQA25DDataset
from liqa_mrgan3d.models.pix2pix_25d import Generator25D, PatchDiscriminator, weights_init_normal
from liqa_mrgan3d.trainer.losses import GANLoss, GaussianBlur2D
from liqa_mrgan3d.utils.config import ensure_dir


class LiQATrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.device = torch.device("cuda" if config.get("cuda", True) and torch.cuda.is_available() else "cpu")
        self.output_dir = ensure_dir(config.get("save_root", "outputs"))
        self.checkpoint_dir = ensure_dir(Path(self.output_dir) / "checkpoints")
        self.log_dir = ensure_dir(Path(self.output_dir) / "logs")

        self.input_nc = int(config.get("input_nc", len(config.get("input_modalities", [])) * config.get("slice_window", 3)))
        self.output_nc = int(config.get("output_nc", 1))
        self.condition_channels_for_d = int(config.get("condition_channels_for_d", self.input_nc))

        self.net_g = Generator25D(
            input_nc=self.input_nc,
            output_nc=self.output_nc,
            base_channels=int(config.get("g_base_channels", 64)),
        ).to(self.device)
        self.net_d = PatchDiscriminator(
            input_nc=self.condition_channels_for_d + self.output_nc,
            ndf=int(config.get("d_base_channels", 64)),
            n_layers=int(config.get("d_layers", 3)),
        ).to(self.device)
        self.net_g.apply(weights_init_normal)
        self.net_d.apply(weights_init_normal)

        self.gan_loss = GANLoss().to(self.device)
        self.l1_loss = nn.L1Loss()
        self.blur = GaussianBlur2D(channels=self.output_nc).to(self.device)

        lr = float(config.get("lr", 1e-4))
        self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(self.net_d.parameters(), lr=lr, betas=(0.5, 0.999))

        self.train_loader = self._make_loader("train", shuffle=True)
        self.val_loader = self._make_loader("val", shuffle=False) if config.get("val_txt_path") else None
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def _make_loader(self, split: str, shuffle: bool) -> DataLoader[dict[str, Any]]:
        dataset = LiQA25DDataset(self.config, split=split)
        return DataLoader(
            dataset,
            batch_size=int(self.config.get("batch_size", self.config.get("batchSize", 1))),
            shuffle=shuffle,
            num_workers=int(self.config.get("n_cpu", 0)),
            pin_memory=self.device.type == "cuda",
        )

    def _condition_for_discriminator(self, real_a: torch.Tensor) -> torch.Tensor:
        if self.condition_channels_for_d == real_a.shape[1]:
            return real_a
        return real_a[:, : self.condition_channels_for_d]

    def train(self) -> None:
        n_epochs = int(self.config.get("n_epochs", 80))
        start_epoch = int(self.config.get("epoch", 0))
        global_step = 0
        for epoch in range(start_epoch, n_epochs):
            self.net_g.train()
            self.net_d.train()
            progress = tqdm(self.train_loader, desc=f"epoch {epoch + 1}/{n_epochs}")
            for batch in progress:
                real_a = batch["A"].to(self.device, non_blocking=True)
                real_b = batch["B"].to(self.device, non_blocking=True)
                cond = self._condition_for_discriminator(real_a)

                # Generator update
                self.optimizer_g.zero_grad(set_to_none=True)
                fake_b = self.net_g(real_a)
                pred_fake = self.net_d(torch.cat([cond, fake_b], dim=1))
                loss_g_gan = self.gan_loss(pred_fake, True) * float(self.config.get("adv_lambda", 1.0))
                loss_g_l1 = self.l1_loss(fake_b, real_b) * float(self.config.get("l1_lambda", 10.0))
                loss_g_blur = self.l1_loss(self.blur(fake_b), self.blur(real_b)) * float(
                    self.config.get("blur_lambda", 0.0)
                )
                loss_g = loss_g_gan + loss_g_l1 + loss_g_blur
                loss_g.backward()
                self.optimizer_g.step()

                # Discriminator update
                self.optimizer_d.zero_grad(set_to_none=True)
                pred_real = self.net_d(torch.cat([cond, real_b], dim=1))
                pred_fake_detached = self.net_d(torch.cat([cond, fake_b.detach()], dim=1))
                loss_d = 0.5 * (self.gan_loss(pred_real, True) + self.gan_loss(pred_fake_detached, False))
                loss_d.backward()
                self.optimizer_d.step()

                if global_step % int(self.config.get("log_interval", 20)) == 0:
                    self.writer.add_scalar("train/loss_g", loss_g.item(), global_step)
                    self.writer.add_scalar("train/loss_d", loss_d.item(), global_step)
                    self.writer.add_scalar("train/loss_g_l1", loss_g_l1.item(), global_step)
                    self.writer.add_scalar("train/loss_g_gan", loss_g_gan.item(), global_step)
                    if loss_g_blur.detach().item() != 0:
                        self.writer.add_scalar("train/loss_g_blur", loss_g_blur.item(), global_step)

                progress.set_postfix(loss_g=f"{loss_g.item():.4f}", loss_d=f"{loss_d.item():.4f}")
                global_step += 1

            self.save_checkpoint("latest", epoch)
            if (epoch + 1) % int(self.config.get("checkpoint_interval", 5)) == 0:
                self.save_checkpoint(f"epoch_{epoch + 1:04d}", epoch)

        self.writer.close()

    def save_checkpoint(self, name: str, epoch: int) -> None:
        path = Path(self.checkpoint_dir) / f"netG_{name}.pt"
        torch.save(
            {
                "epoch": epoch,
                "config": self.config,
                "net_g": self.net_g.state_dict(),
                "net_d": self.net_d.state_dict(),
                "optimizer_g": self.optimizer_g.state_dict(),
                "optimizer_d": self.optimizer_d.state_dict(),
            },
            path,
        )
