"""3D MrGAN trainer.

Algorithmically mirrors ``reference/MrGAN/trainer/p2pTrainer_v6.py:97-189`` with
the only deviation being 3D adaptations (Conv3d, trilinear interpolation,
slice-wise LPIPS, 3D smoothness regulariser). The training step per batch is::

    fake_B    = G(real_A)
    flow      = Reg(fake_B, real_B)
    reg_B     = Transformer(fake_B, flow)
    loss_L1   = p2p_lambda  * L1(reg_B, real_B)
    loss_GAN  = adv_lambda  * sum_i MSE(D_i([real_A_i, reg_B]), 1)
    loss_blur = blur_lambda * L1(blur(reg_B), blur(real_B))
    loss_perc = perc_lambda * LPIPS_2D_slicewise(reg_B, real_B)
    loss_SM   = corr_lambda  * L1(reg_B, real_B)            # registration anchor
    loss_SR   = smooth_lambda* SmoothnessLoss3D(flow)
    loss_shape= shape_lambda * SoftDice3D(sigmoid(UNet(reg_B)), onehot(M))
    -> total_G.backward(); step(opt_G, opt_R)

then refresh fake_B and update each discriminator with the usual real/fake MSE.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from liqa_mrgan3d.data.datasets_liqa import LiQA3DPatchDataset
from liqa_mrgan3d.models.mrgan3d import (
    MrGANDiscriminator3D,
    MrGANGenerator3D,
    mrgan_weights_init_normal,
)
from liqa_mrgan3d.models.reg3d import Reg3D
from liqa_mrgan3d.models.transformer3d import Transformer3D
from liqa_mrgan3d.models.unet3d import UNet3D
from liqa_mrgan3d.trainer.losses import (
    GANLoss,
    GaussianBlur3D,
    LPIPS2DSliceWise,
    SmoothnessLoss3D,
    SoftDice3D,
    mask_to_onehot_3d,
)
from liqa_mrgan3d.utils.config import ensure_dir


class LiQA3DTrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.device = torch.device(
            "cuda" if config.get("cuda", True) and torch.cuda.is_available() else "cpu"
        )
        self.output_dir = ensure_dir(config.get("save_root", "outputs/liqa_mrgan3d"))
        self.checkpoint_dir = ensure_dir(Path(self.output_dir) / "checkpoints")
        self.log_dir = ensure_dir(Path(self.output_dir) / "logs")

        self.input_nc = int(
            config.get("input_nc", len(config.get("input_modalities", ["T1", "T2", "DWI_800"])))
        )
        self.output_nc = int(config.get("output_nc", 1))
        self.use_regist = bool(config.get("regist", True))
        self.use_shape = bool(config.get("shape_loss", True))
        self.palette = config.get("palette", [[0], [1]])
        self.num_classes = int(config.get("num_classes", 2))

        # Generator + N discriminators (one per input channel)
        self.net_g = MrGANGenerator3D(
            in_c=self.input_nc,
            mid_c=int(config.get("g_base_channels", 32)),
            layers=int(config.get("g_layers", 2)),
            s_layers=int(config.get("g_share_layers", 3)),
        ).to(self.device)
        self.net_g.apply(mrgan_weights_init_normal)

        self.net_d = nn.ModuleList(
            [
                MrGANDiscriminator3D(
                    input_nc=1 + self.output_nc,
                    ndf=int(config.get("d_base_channels", 32)),
                    n_layers=int(config.get("d_layers", 3)),
                ).to(self.device)
                for _ in range(self.input_nc)
            ]
        )
        self.net_d.apply(mrgan_weights_init_normal)

        # Registration sub-network + spatial transformer.
        if self.use_regist:
            self.reg = Reg3D(in_channels_a=self.output_nc, in_channels_b=self.output_nc).to(self.device)
            self.transformer = Transformer3D().to(self.device)
        else:
            self.reg = None
            self.transformer = None

        # Pretrained 3D UNet for the shape loss (frozen).
        if self.use_shape:
            self.unet = UNet3D(
                img_ch=1,
                num_classes=self.num_classes,
                depth=int(config.get("unet_depth", 1)),
                base_channels=int(config.get("unet_base_channels", 32)),
            ).to(self.device)
            unet_chk = config.get("unet_chk")
            if unet_chk and Path(unet_chk).exists():
                state = torch.load(unet_chk, map_location=self.device)
                self.unet.load_state_dict(state)
                print(f"loaded UNet checkpoint from {unet_chk}")
            else:
                print(
                    f"[warning] unet_chk {unet_chk!r} not found; shape loss will use an untrained UNet."
                )
            for p in self.unet.parameters():
                p.requires_grad = False
            self.unet.eval()
        else:
            self.unet = None

        # Losses.
        self.gan_loss = GANLoss().to(self.device)
        self.l1_loss = nn.L1Loss()
        self.blur = GaussianBlur3D(channels=self.output_nc).to(self.device)
        self.smooth_loss = SmoothnessLoss3D().to(self.device)
        self.soft_dice = SoftDice3D(self.num_classes).to(self.device)
        self.lpips: LPIPS2DSliceWise | None = None
        if float(config.get("perceptual_lambda", 0.0)) > 0:
            self.lpips = LPIPS2DSliceWise(net="vgg").to(self.device)

        # Optimisers.
        lr = float(config.get("lr", 1e-4))
        self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(
            itertools.chain(*[d.parameters() for d in self.net_d]),
            lr=lr,
            betas=(0.5, 0.999),
        )
        if self.use_regist:
            self.optimizer_r = torch.optim.Adam(self.reg.parameters(), lr=lr, betas=(0.5, 0.999))
        else:
            self.optimizer_r = None

        # Data.
        self.train_loader = self._make_loader("train", shuffle=True)
        self.val_loader = (
            self._make_loader("val", shuffle=False) if config.get("val_txt_path") else None
        )

        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    # ------------------------------------------------------------------ data
    def _make_loader(self, split: str, shuffle: bool) -> DataLoader[dict[str, Any]]:
        dataset = LiQA3DPatchDataset(self.config, split=split)
        return DataLoader(
            dataset,
            batch_size=int(self.config.get("batch_size", 1)),
            shuffle=shuffle,
            num_workers=int(self.config.get("n_cpu", 0)),
            pin_memory=self.device.type == "cuda",
        )

    # --------------------------------------------------------------- training
    def train(self) -> None:
        n_epochs = int(self.config.get("n_epochs", 80))
        start_epoch = int(self.config.get("epoch", 0))
        log_interval = int(self.config.get("log_interval", 10))
        adv_lambda = float(self.config.get("adv_lambda", 1.0))
        p2p_lambda = float(self.config.get("p2p_lambda", 5.0))
        blur_lambda = float(self.config.get("blur_lambda", 5.0))
        perc_lambda = float(self.config.get("perceptual_lambda", 0.5))
        corr_lambda = float(self.config.get("corr_lambda", 1.0))
        smooth_lambda = float(self.config.get("smooth_lambda", 1.0))
        shape_lambda = float(self.config.get("shape_lambda", 1.0))

        global_step = 0
        for epoch in range(start_epoch, n_epochs):
            self.net_g.train()
            for d in self.net_d:
                d.train()
            if self.reg is not None:
                self.reg.train()
            progress = tqdm(self.train_loader, desc=f"3d epoch {epoch + 1}/{n_epochs}")
            for batch in progress:
                real_a = batch["A"].to(self.device, non_blocking=True)
                real_b = batch["B"].to(self.device, non_blocking=True)
                mask = batch.get("M")
                if mask is not None:
                    mask = mask.to(self.device, non_blocking=True)

                # ----- Generator + Reg update -----
                self.optimizer_g.zero_grad(set_to_none=True)
                if self.optimizer_r is not None:
                    self.optimizer_r.zero_grad(set_to_none=True)

                fake_b = self.net_g(real_a)

                if self.reg is not None and self.transformer is not None:
                    flow = self.reg(fake_b, real_b)
                    reg_b = self.transformer(fake_b, flow)
                else:
                    flow = None
                    reg_b = fake_b

                loss_l1 = self.l1_loss(reg_b, real_b) * p2p_lambda

                loss_gan = torch.zeros((), device=self.device)
                for i in range(self.input_nc):
                    fake_pair = torch.cat([real_a[:, i : i + 1], reg_b], dim=1)
                    pred_fake = self.net_d[i](fake_pair)
                    loss_gan = loss_gan + self.gan_loss(pred_fake, True) * adv_lambda

                blurred_pred = self.blur(reg_b)
                blurred_real = self.blur(real_b)
                loss_blur = self.l1_loss(blurred_pred, blurred_real) * blur_lambda

                if self.lpips is not None:
                    loss_perc = self.lpips(reg_b, real_b) * perc_lambda
                else:
                    loss_perc = torch.zeros((), device=self.device)

                if flow is not None:
                    loss_sm = self.l1_loss(reg_b, real_b) * corr_lambda
                    loss_sr = self.smooth_loss(flow) * smooth_lambda
                else:
                    loss_sm = torch.zeros((), device=self.device)
                    loss_sr = torch.zeros((), device=self.device)

                if self.unet is not None and mask is not None:
                    regist_seg = torch.sigmoid(self.unet(reg_b))
                    real_seg_onehot = mask_to_onehot_3d(mask.long().cpu(), self.palette).to(
                        self.device
                    )
                    loss_shape = self.soft_dice(regist_seg, real_seg_onehot) * shape_lambda
                else:
                    loss_shape = torch.zeros((), device=self.device)

                total_g = loss_l1 + loss_gan + loss_blur + loss_perc + loss_sm + loss_sr + loss_shape
                total_g.backward()
                self.optimizer_g.step()
                if self.optimizer_r is not None:
                    self.optimizer_r.step()

                # ----- Discriminator update -----
                self.optimizer_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    fake_b_d = self.net_g(real_a)
                loss_d_total = torch.zeros((), device=self.device)
                for i in range(self.input_nc):
                    real_pair = torch.cat([real_a[:, i : i + 1], real_b], dim=1)
                    fake_pair = torch.cat([real_a[:, i : i + 1], fake_b_d], dim=1)
                    pred_real = self.net_d[i](real_pair)
                    pred_fake = self.net_d[i](fake_pair)
                    loss_d_total = (
                        loss_d_total
                        + (self.gan_loss(pred_real, True) + self.gan_loss(pred_fake, False)) * adv_lambda
                    )
                loss_d_total.backward()
                self.optimizer_d.step()

                if global_step % log_interval == 0:
                    self.writer.add_scalar("train/loss_g", total_g.item(), global_step)
                    self.writer.add_scalar("train/loss_d", loss_d_total.item(), global_step)
                    self.writer.add_scalar("train/loss_l1", loss_l1.item(), global_step)
                    self.writer.add_scalar("train/loss_gan", loss_gan.item(), global_step)
                    self.writer.add_scalar("train/loss_blur", loss_blur.item(), global_step)
                    self.writer.add_scalar("train/loss_perc", loss_perc.item(), global_step)
                    self.writer.add_scalar("train/loss_sm", loss_sm.item(), global_step)
                    self.writer.add_scalar("train/loss_sr", loss_sr.item(), global_step)
                    self.writer.add_scalar("train/loss_shape", loss_shape.item(), global_step)

                progress.set_postfix(
                    g=f"{total_g.item():.3f}",
                    d=f"{loss_d_total.item():.3f}",
                    l1=f"{loss_l1.item():.3f}",
                )
                global_step += 1

            self.save_checkpoint("latest", epoch)
            if (epoch + 1) % int(self.config.get("checkpoint_interval", 5)) == 0:
                self.save_checkpoint(f"epoch_{epoch + 1:04d}", epoch)

        self.writer.close()

    # ----------------------------------------------------------- checkpoints
    def save_checkpoint(self, name: str, epoch: int) -> None:
        path = Path(self.checkpoint_dir) / f"netG_{name}.pt"
        state: dict[str, Any] = {
            "epoch": epoch,
            "config": self.config,
            "net_g": self.net_g.state_dict(),
            "net_d": [d.state_dict() for d in self.net_d],
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
        }
        if self.reg is not None:
            state["reg"] = self.reg.state_dict()
            state["optimizer_r"] = self.optimizer_r.state_dict()
        torch.save(state, path)
