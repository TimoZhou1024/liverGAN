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

Two training modes, selected by ``config['mode']``:

* ``3d_patch`` (default): patch-sampled training, single-GPU or DDP/FSDP.
* ``3d_full``: whole-volume training, designed for FSDP across multiple GPUs
  on a server. Under FSDP the generator and frozen UNet are FULL_SHARDed
  across ranks so parameter/optimizer memory is pooled; the smaller
  discriminators and ``Reg3D`` use ``SHARD_GRAD_OP`` to keep forward fast.
  Activation checkpointing is applied to the heavy ``ConvBlock3D`` and
  ``EncoderBlock3D`` modules.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from liqa_mrgan3d.data.datasets_liqa import LiQA3DFullVolumeDataset, LiQA3DPatchDataset
from liqa_mrgan3d.models.mrgan3d import (
    ConvBlock3D,
    MrGANDiscriminator3D,
    MrGANGenerator3D,
    mrgan_weights_init_normal,
)
from liqa_mrgan3d.models.reg3d import ConvAct3D, Reg3D
from liqa_mrgan3d.models.transformer3d import Transformer3D
from liqa_mrgan3d.models.unet3d import EncoderBlock3D, UNet3D
from liqa_mrgan3d.trainer.distributed import (
    apply_activation_checkpoint,
    fsdp_wrap,
    gather_full_state_dict,
    init_distributed,
)
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
        self.mode = str(config.get("mode", "3d_patch"))

        # Distributed init must run before CUDA device allocation so the
        # per-rank device index is set correctly.
        self.is_dist, self.rank, self.world_size = init_distributed()
        self.is_rank0 = self.rank == 0

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
        self.use_amp = bool(config.get("amp", False))
        self.use_activation_checkpoint = bool(config.get("activation_checkpoint", False))

        # --- Generator + N discriminators (one per input channel) ---
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

        # --- Registration sub-network + spatial transformer ---
        if self.use_regist:
            self.reg = Reg3D(in_channels_a=self.output_nc, in_channels_b=self.output_nc).to(
                self.device
            )
            self.transformer = Transformer3D().to(self.device)
        else:
            self.reg = None
            self.transformer = None

        # --- Frozen 3D UNet for the shape loss ---
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
                if self.is_rank0:
                    print(f"loaded UNet checkpoint from {unet_chk}")
            elif self.is_rank0:
                print(
                    f"[warning] unet_chk {unet_chk!r} not found; shape loss will use an "
                    "untrained UNet."
                )
            for p in self.unet.parameters():
                p.requires_grad = False
            self.unet.eval()
        else:
            self.unet = None

        # --- FSDP wrapping (only under torchrun) ---
        if self.is_dist:
            from torch.distributed.fsdp import ShardingStrategy

            # SyncBatchNorm inside discriminators so BN stats aggregate across ranks.
            for i, d in enumerate(self.net_d):
                self.net_d[i] = nn.SyncBatchNorm.convert_sync_batchnorm(d)
            self.net_g = fsdp_wrap(
                self.net_g,
                min_params=10_000_000,
                sharding=ShardingStrategy.FULL_SHARD,
                checkpoint=self.use_activation_checkpoint,
                checkpoint_target_cls=(ConvBlock3D,),
                use_amp=self.use_amp,
            )
            self.net_d = nn.ModuleList(
                [
                    fsdp_wrap(
                        d,
                        min_params=5_000_000,
                        sharding=ShardingStrategy.SHARD_GRAD_OP,
                        checkpoint=False,
                        use_amp=self.use_amp,
                    )
                    for d in self.net_d
                ]
            )
            if self.reg is not None:
                self.reg = fsdp_wrap(
                    self.reg,
                    min_params=5_000_000,
                    sharding=ShardingStrategy.SHARD_GRAD_OP,
                    checkpoint=self.use_activation_checkpoint,
                    checkpoint_target_cls=(ConvAct3D,),
                    use_amp=self.use_amp,
                )
            if self.unet is not None:
                # Frozen → NO_SHARD keeps params local and avoids FSDP no_grad issues.
                # Activation checkpoint on the big encoder blocks: the shape loss's
                # gradient has to flow through UNet back to reg_b, so UNet activations
                # are stored during backward even though its params are frozen.
                self.unet = fsdp_wrap(
                    self.unet,
                    min_params=10_000_000,
                    sharding=ShardingStrategy.NO_SHARD,
                    checkpoint=self.use_activation_checkpoint,
                    checkpoint_target_cls=(EncoderBlock3D,),
                    use_amp=False,  # Keep UNet in fp32 — it's a thin inference pass.
                )
        elif self.use_activation_checkpoint:
            # Single-GPU path: still apply activation checkpointing so the [32, 384, 384]
            # volume fits on one card. Same target modules as the FSDP branch above.
            apply_activation_checkpoint(self.net_g, (ConvBlock3D,))
            if self.reg is not None:
                apply_activation_checkpoint(self.reg, (ConvAct3D,))
            if self.unet is not None:
                apply_activation_checkpoint(self.unet, (EncoderBlock3D,))

        # --- Losses ---
        self.gan_loss = GANLoss().to(self.device)
        self.l1_loss = nn.L1Loss()
        self.blur = GaussianBlur3D(channels=self.output_nc).to(self.device)
        self.smooth_loss = SmoothnessLoss3D().to(self.device)
        self.soft_dice = SoftDice3D(self.num_classes).to(self.device)
        self.lpips: LPIPS2DSliceWise | None = None
        if float(config.get("perceptual_lambda", 0.0)) > 0:
            # `lpips_slices` subsamples k slices per call to cap VGG activation memory.
            # Falls back to all slices when unset.
            lpips_slices = config.get("lpips_slices")
            self.lpips = LPIPS2DSliceWise(
                net="vgg",
                num_slices=int(lpips_slices) if lpips_slices else None,
            ).to(self.device)

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

        self.scaler_g: torch.cuda.amp.GradScaler | None = (
            torch.cuda.amp.GradScaler() if self.use_amp and self.device.type == "cuda" else None
        )
        self.scaler_d: torch.cuda.amp.GradScaler | None = (
            torch.cuda.amp.GradScaler() if self.use_amp and self.device.type == "cuda" else None
        )

        # --- Data ---
        self.train_loader, self.train_sampler = self._make_loader("train", shuffle=True)
        if config.get("val_txt_path"):
            self.val_loader, _ = self._make_loader("val", shuffle=False)
        else:
            self.val_loader = None

        self.writer = (
            SummaryWriter(log_dir=str(self.log_dir)) if self.is_rank0 else None
        )

    # ------------------------------------------------------------------ data
    def _make_dataset(self, split: str) -> Dataset[dict[str, Any]]:
        if self.mode == "3d_full":
            return LiQA3DFullVolumeDataset(self.config, split=split)
        return LiQA3DPatchDataset(self.config, split=split)

    def _make_loader(
        self, split: str, shuffle: bool
    ) -> tuple[DataLoader[dict[str, Any]], DistributedSampler | None]:
        dataset = self._make_dataset(split)
        sampler: DistributedSampler | None = None
        if self.is_dist:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                drop_last=True,
            )
        loader = DataLoader(
            dataset,
            batch_size=int(self.config.get("batch_size", 1)),
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=int(self.config.get("n_cpu", 0)),
            pin_memory=self.device.type == "cuda",
        )
        return loader, sampler

    # --------------------------------------------------------------- training
    def train(self) -> None:
        n_epochs = int(self.config.get("n_epochs", 80))
        start_epoch = int(self.config.get("epoch", 0))
        log_interval = int(self.config.get("log_interval", 10))
        adv_lambda = float(self.config.get("adv_lambda", 1.0))
        p2p_lambda = float(self.config.get("p2p_lambda", 5.0))
        # Direct L1 supervision on G(A) before Reg warps it. Without this term Reg
        # acts as a shortcut: it can warp G's mean-image output to match B, so G
        # never learns input→output alignment. See Phase-1 diagnostic on epoch=74
        # ckpt: L1(fake_b, B)=0.74 vs L1(reg_b, B)=0.007.
        direct_lambda = float(self.config.get("direct_lambda", 0.0))
        mag_lambda = float(self.config.get("mag_lambda", 0.0))
        reg_warmup_epochs = int(self.config.get("reg_warmup_epochs", 0))
        blur_lambda = float(self.config.get("blur_lambda", 5.0))
        perc_lambda = float(self.config.get("perceptual_lambda", 0.5))
        corr_lambda = float(self.config.get("corr_lambda", 1.0))
        smooth_lambda = float(self.config.get("smooth_lambda", 1.0))
        shape_lambda = float(self.config.get("shape_lambda", 1.0))

        global_step = 0
        for epoch in range(start_epoch, n_epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self.net_g.train()
            for d in self.net_d:
                d.train()
            if self.reg is not None:
                self.reg.train()

            if self.is_rank0:
                progress = tqdm(
                    self.train_loader, desc=f"{self.mode} epoch {epoch + 1}/{n_epochs}"
                )
            else:
                progress = self.train_loader

            for batch in progress:
                real_a = batch["A"].to(self.device, non_blocking=True)
                real_b = batch["B"].to(self.device, non_blocking=True)
                mask = batch.get("M")
                if mask is not None:
                    mask = mask.to(self.device, non_blocking=True)

                total_g, loss_dict, fake_b_cache = self._g_step(
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

                if self.is_rank0 and global_step % log_interval == 0 and self.writer is not None:
                    self.writer.add_scalar("train/loss_g", total_g, global_step)
                    self.writer.add_scalar("train/loss_d", loss_d_total, global_step)
                    for name, value in loss_dict.items():
                        self.writer.add_scalar(f"train/{name}", value, global_step)

                if self.is_rank0 and isinstance(progress, tqdm):
                    progress.set_postfix(
                        g=f"{total_g:.3f}",
                        d=f"{loss_d_total:.3f}",
                        l1=f"{loss_dict['loss_l1']:.3f}",
                    )
                global_step += 1

            self.save_checkpoint("latest", epoch)
            if (epoch + 1) % int(self.config.get("checkpoint_interval", 5)) == 0:
                self.save_checkpoint(f"epoch_{epoch + 1:04d}", epoch)

        if self.writer is not None:
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
            else _nullcontext()
        )

        # During warmup, freeze Reg as identity so G must learn input→output
        # alignment by itself. After warmup, Reg is allowed to compensate for
        # residual misalignment between G(A) and B.
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

            # Direct L1 on G(A) — the primary fix for the Reg-shortcut failure
            # mode (G learning a single mean GED4 image while Reg compensates
            # per-sample). Always computed; lambda may be 0 to disable.
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
                # Penalise raw flow magnitude so Reg pays for any displacement,
                # not just non-smooth ones. Without this, smoothness alone allows
                # 8-voxel uniform shifts (observed in epoch=74 ckpt).
                loss_mag = flow.abs().mean() * mag_lambda
            else:
                loss_sm = torch.zeros((), device=self.device)
                loss_sr = torch.zeros((), device=self.device)
                loss_mag = torch.zeros((), device=self.device)

            if self.unet is not None and mask is not None:
                # UNet params are frozen (requires_grad=False), so no grads flow into
                # UNet weights. But reg_b DOES require grad → UNet's forward still
                # has to record activations so its backward can propagate dL/dreg_b.
                # Activation checkpointing on the UNet's EncoderBlock3D's (wired in
                # __init__) is what keeps this tractable for [32, 384, 384] inputs.
                regist_seg = torch.sigmoid(self.unet(reg_b))
                real_seg_onehot = mask_to_onehot_3d(mask.long(), self.palette)
                loss_shape = self.soft_dice(regist_seg, real_seg_onehot) * shape_lambda
            else:
                loss_shape = torch.zeros((), device=self.device)

        # LPIPS is evaluated in fp32 outside autocast — VGG has fp16 instabilities.
        if self.lpips is not None:
            loss_perc = self.lpips(reg_b.float(), real_b.float()) * perc_lambda
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
    def _d_step(self, real_a: torch.Tensor, real_b: torch.Tensor, *, adv_lambda: float) -> float:
        self.optimizer_d.zero_grad(set_to_none=True)

        amp_ctx = (
            torch.cuda.amp.autocast(dtype=torch.float16)
            if self.use_amp and self.device.type == "cuda"
            else _nullcontext()
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
        g_state = gather_full_state_dict(self.net_g)
        d_states = [gather_full_state_dict(d) for d in self.net_d]
        reg_state = gather_full_state_dict(self.reg) if self.reg is not None else None

        if not self.is_rank0:
            return

        path = Path(self.checkpoint_dir) / f"netG_{name}.pt"
        state: dict[str, Any] = {
            "epoch": epoch,
            "config": self.config,
            "net_g": g_state,
            "net_d": d_states,
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
        }
        if reg_state is not None:
            state["reg"] = reg_state
            assert self.optimizer_r is not None
            state["optimizer_r"] = self.optimizer_r.state_dict()
        torch.save(state, path)


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
