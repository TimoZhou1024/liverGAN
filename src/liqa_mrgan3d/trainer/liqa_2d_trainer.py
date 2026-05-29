"""2D MrGAN trainer.

Algorithmically identical to :class:`LiQA3DTrainer` (same losses, same
ordering, same Reg-shortcut fixes — direct_lambda / mag_lambda /
reg_warmup_epochs) but operates on 2D liver-bearing slices via
:class:`LiQA2DSliceDataset`.

No FSDP / activation-checkpoint plumbing here: 2D slices are small enough
(``[3, 256, 256]`` per sample) that batch sizes of 8-32 fit on a single
modern GPU, so we keep the loop simple. AMP can still be enabled via the
``amp`` config field.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from liqa_mrgan3d.data.datasets_liqa import LiQA2DSliceDataset
from liqa_mrgan3d.models.mrgan2d import (
    MrGANDiscriminator2D,
    MrGANGenerator2D,
    mrgan_weights_init_normal_2d,
)
from liqa_mrgan3d.models.reg2d import Reg2D
from liqa_mrgan3d.models.transformer2d import Transformer2D
from liqa_mrgan3d.models.unet2d import UNet2D
from liqa_mrgan3d.trainer.losses import (
    GANLoss,
    GaussianBlur2D,
    SmoothnessLoss2D,
    SoftDice3D,  # dim-agnostic — works on 2D too
    mask_to_onehot_2d,
)
from liqa_mrgan3d.utils.config import ensure_dir


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class LiQA2DTrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.mode = str(config.get("mode", "2d_slice"))

        self.device = torch.device(
            "cuda" if config.get("cuda", True) and torch.cuda.is_available() else "cpu"
        )
        self.output_dir = ensure_dir(config.get("save_root", "outputs/liqa_mrgan2d"))
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
        self.use_amp = bool(config.get("amp", False))

        # --- Generator + N PatchGAN discriminators (one per input modality) ---
        self.net_g = MrGANGenerator2D(
            in_c=self.input_nc,
            mid_c=int(config.get("g_base_channels", 64)),
            layers=int(config.get("g_layers", 2)),
            s_layers=int(config.get("g_share_layers", 3)),
        ).to(self.device)
        self.net_g.apply(mrgan_weights_init_normal_2d)

        self.net_d = nn.ModuleList(
            [
                MrGANDiscriminator2D(
                    input_nc=1 + self.output_nc,
                    ndf=int(config.get("d_base_channels", 64)),
                    n_layers=int(config.get("d_layers", 3)),
                ).to(self.device)
                for _ in range(self.input_nc)
            ]
        )
        self.net_d.apply(mrgan_weights_init_normal_2d)

        # --- Registration sub-network + spatial transformer ---
        if self.use_regist:
            self.reg = Reg2D(in_channels_a=self.output_nc, in_channels_b=self.output_nc).to(
                self.device
            )
            self.transformer = Transformer2D().to(self.device)
        else:
            self.reg = None
            self.transformer = None

        # --- Frozen 2D UNet for the shape loss ---
        if self.use_shape:
            self.unet = UNet2D(
                img_ch=1,
                num_classes=self.num_classes,
                depth=int(config.get("unet_depth", 1)),
                base_channels=int(config.get("unet_base_channels", 32)),
            ).to(self.device)
            unet_chk = config.get("unet_chk")
            if unet_chk and Path(unet_chk).exists():
                state = torch.load(unet_chk, map_location=self.device)
                self.unet.load_state_dict(state)
                print(f"loaded UNet2D checkpoint from {unet_chk}")
            else:
                print(
                    f"[warning] unet_chk {unet_chk!r} not found; shape loss will use an "
                    "untrained UNet."
                )
            for p in self.unet.parameters():
                p.requires_grad = False
            self.unet.eval()
        else:
            self.unet = None

        # --- Losses ---
        self.gan_loss = GANLoss().to(self.device)
        self.l1_loss = nn.L1Loss()
        self.blur = GaussianBlur2D(channels=self.output_nc).to(self.device)
        self.smooth_loss = SmoothnessLoss2D().to(self.device)
        self.soft_dice = SoftDice3D(self.num_classes).to(self.device)  # dim-agnostic

        self.lpips = None
        if float(config.get("perceptual_lambda", 0.0)) > 0:
            import lpips  # heavy; only import when needed

            self.lpips = lpips.LPIPS(net="vgg").to(self.device)
            for p in self.lpips.parameters():
                p.requires_grad = False

        # --- Optimisers ---
        lr = float(config.get("lr", 1e-4))
        self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(
            itertools.chain(*[d.parameters() for d in self.net_d]),
            lr=lr,
            betas=(0.5, 0.999),
        )
        if self.use_regist:
            self.optimizer_r = torch.optim.Adam(
                self.reg.parameters(), lr=lr, betas=(0.5, 0.999)
            )
        else:
            self.optimizer_r = None

        self.scaler_g = (
            torch.cuda.amp.GradScaler() if self.use_amp and self.device.type == "cuda" else None
        )
        self.scaler_d = (
            torch.cuda.amp.GradScaler() if self.use_amp and self.device.type == "cuda" else None
        )

        # --- Data ---
        self.train_loader = self._make_loader("train", shuffle=True)
        self.val_loader = (
            self._make_loader("val", shuffle=False) if config.get("val_txt_path") else None
        )

        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    # ------------------------------------------------------------------ data
    def _make_dataset(self, split: str) -> Dataset[dict[str, Any]]:
        return LiQA2DSliceDataset(self.config, split=split)

    def _make_loader(self, split: str, shuffle: bool) -> DataLoader[dict[str, Any]]:
        dataset = self._make_dataset(split)
        return DataLoader(
            dataset,
            batch_size=int(self.config.get("batch_size", 8)),
            shuffle=shuffle,
            num_workers=int(self.config.get("n_cpu", 0)),
            pin_memory=self.device.type == "cuda",
            drop_last=shuffle,
        )

    # --------------------------------------------------------------- training
    def train(self) -> None:
        n_epochs = int(self.config.get("n_epochs", 80))
        start_epoch = int(self.config.get("epoch", 0))
        log_interval = int(self.config.get("log_interval", 50))
        adv_lambda = float(self.config.get("adv_lambda", 1.0))
        p2p_lambda = float(self.config.get("p2p_lambda", 2.0))
        direct_lambda = float(self.config.get("direct_lambda", 5.0))
        mag_lambda = float(self.config.get("mag_lambda", 2.0))
        reg_warmup_epochs = int(self.config.get("reg_warmup_epochs", 10))
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

            progress = tqdm(self.train_loader, desc=f"{self.mode} epoch {epoch + 1}/{n_epochs}")
            for batch in progress:
                real_a = batch["A"].to(self.device, non_blocking=True)
                real_b = batch["B"].to(self.device, non_blocking=True)
                mask = batch.get("M")
                if mask is not None:
                    mask = mask.to(self.device, non_blocking=True)

                total_g, loss_dict, _ = self._g_step(
                    real_a,
                    real_b,
                    mask,
                    epoch=epoch,
                    adv_lambda=adv_lambda,
                    p2p_lambda=p2p_lambda,
                    direct_lambda=direct_lambda,
                    mag_lambda=mag_lambda,
                    reg_warmup_epochs=reg_warmup_epochs,
                    blur_lambda=blur_lambda,
                    perc_lambda=perc_lambda,
                    corr_lambda=corr_lambda,
                    smooth_lambda=smooth_lambda,
                    shape_lambda=shape_lambda,
                )
                loss_d_total = self._d_step(real_a, real_b, adv_lambda=adv_lambda)

                if global_step % log_interval == 0:
                    self.writer.add_scalar("train/loss_g", total_g, global_step)
                    self.writer.add_scalar("train/loss_d", loss_d_total, global_step)
                    for name, value in loss_dict.items():
                        self.writer.add_scalar(f"train/{name}", value, global_step)

                progress.set_postfix(
                    g=f"{total_g:.3f}",
                    d=f"{loss_d_total:.3f}",
                    l1d=f"{loss_dict['loss_l1_direct']:.3f}",
                )
                global_step += 1

            self.save_checkpoint("latest", epoch)
            if (epoch + 1) % int(self.config.get("checkpoint_interval", 5)) == 0:
                self.save_checkpoint(f"epoch_{epoch + 1:04d}", epoch)

        self.writer.close()

    # ---------------------------------------------------------------- G step
    def _g_step(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        epoch: int,
        adv_lambda: float,
        p2p_lambda: float,
        direct_lambda: float,
        mag_lambda: float,
        reg_warmup_epochs: int,
        blur_lambda: float,
        perc_lambda: float,
        corr_lambda: float,
        smooth_lambda: float,
        shape_lambda: float,
    ) -> tuple[float, dict[str, float], torch.Tensor]:
        self.optimizer_g.zero_grad(set_to_none=True)
        if self.optimizer_r is not None:
            self.optimizer_r.zero_grad(set_to_none=True)

        amp_ctx = (
            torch.cuda.amp.autocast(dtype=torch.float16)
            if self.use_amp and self.device.type == "cuda"
            else _NullCtx()
        )

        reg_active = (
            self.reg is not None
            and self.transformer is not None
            and epoch >= reg_warmup_epochs
        )

        with amp_ctx:
            fake_b = self.net_g(real_a)

            if reg_active:
                flow = self.reg(fake_b, real_b)
                reg_b = self.transformer(fake_b, flow)
            else:
                flow = None
                reg_b = fake_b

            loss_l1_direct = self.l1_loss(fake_b, real_b) * direct_lambda
            loss_l1 = self.l1_loss(reg_b, real_b) * p2p_lambda

            loss_gan = torch.zeros((), device=self.device)
            for i in range(self.input_nc):
                fake_pair = torch.cat([real_a[:, i : i + 1], reg_b], dim=1)
                pred_fake = self.net_d[i](fake_pair)
                loss_gan = loss_gan + self.gan_loss(pred_fake, True) * adv_lambda

            loss_blur = self.l1_loss(self.blur(reg_b), self.blur(real_b)) * blur_lambda

            if flow is not None:
                loss_sm = self.l1_loss(reg_b, real_b) * corr_lambda
                loss_sr = self.smooth_loss(flow) * smooth_lambda
                loss_mag = flow.abs().mean() * mag_lambda
            else:
                loss_sm = torch.zeros((), device=self.device)
                loss_sr = torch.zeros((), device=self.device)
                loss_mag = torch.zeros((), device=self.device)

            if self.unet is not None and mask is not None:
                regist_seg = torch.sigmoid(self.unet(reg_b))
                real_seg_onehot = mask_to_onehot_2d(mask.long(), self.palette)
                loss_shape = self.soft_dice(regist_seg, real_seg_onehot) * shape_lambda
            else:
                loss_shape = torch.zeros((), device=self.device)

        # LPIPS in fp32 outside autocast (VGG fp16 instability).
        if self.lpips is not None:
            # LPIPS expects 3-channel [-1, 1] images. reg_b/real_b are 1-channel.
            pred_3c = reg_b.float().clamp(-1, 1).repeat(1, 3, 1, 1)
            target_3c = real_b.float().clamp(-1, 1).repeat(1, 3, 1, 1)
            loss_perc = self.lpips(pred_3c, target_3c).mean() * perc_lambda
        else:
            loss_perc = torch.zeros((), device=self.device)

        total_g = (
            loss_l1
            + loss_l1_direct
            + loss_gan
            + loss_blur
            + loss_perc
            + loss_sm
            + loss_sr
            + loss_mag
            + loss_shape
        )

        if self.scaler_g is not None:
            self.scaler_g.scale(total_g).backward()
            self.scaler_g.step(self.optimizer_g)
            if self.optimizer_r is not None:
                self.scaler_g.step(self.optimizer_r)
            self.scaler_g.update()
        else:
            total_g.backward()
            self.optimizer_g.step()
            if self.optimizer_r is not None:
                self.optimizer_r.step()

        loss_dict = {
            "loss_l1": float(loss_l1.detach()),
            "loss_l1_direct": float(loss_l1_direct.detach()),
            "loss_gan": float(loss_gan.detach()),
            "loss_blur": float(loss_blur.detach()),
            "loss_perc": float(loss_perc.detach()),
            "loss_sm": float(loss_sm.detach()),
            "loss_sr": float(loss_sr.detach()),
            "loss_mag": float(loss_mag.detach()),
            "loss_shape": float(loss_shape.detach()),
        }
        return float(total_g.detach()), loss_dict, fake_b.detach()

    # ---------------------------------------------------------------- D step
    def _d_step(
        self, real_a: torch.Tensor, real_b: torch.Tensor, *, adv_lambda: float
    ) -> float:
        self.optimizer_d.zero_grad(set_to_none=True)

        amp_ctx = (
            torch.cuda.amp.autocast(dtype=torch.float16)
            if self.use_amp and self.device.type == "cuda"
            else _NullCtx()
        )

        with amp_ctx:
            with torch.no_grad():
                fake_b_d = self.net_g(real_a)
            loss_d_total = torch.zeros((), device=self.device)
            for i in range(self.input_nc):
                real_pair = torch.cat([real_a[:, i : i + 1], real_b], dim=1)
                fake_pair = torch.cat([real_a[:, i : i + 1], fake_b_d], dim=1)
                pred_real = self.net_d[i](real_pair)
                pred_fake = self.net_d[i](fake_pair)
                loss_d_total = loss_d_total + (
                    self.gan_loss(pred_real, True) + self.gan_loss(pred_fake, False)
                ) * adv_lambda

        if self.scaler_d is not None:
            self.scaler_d.scale(loss_d_total).backward()
            self.scaler_d.step(self.optimizer_d)
            self.scaler_d.update()
        else:
            loss_d_total.backward()
            self.optimizer_d.step()
        return float(loss_d_total.detach())

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
            assert self.optimizer_r is not None
            state["optimizer_r"] = self.optimizer_r.state_dict()
        torch.save(state, path)
