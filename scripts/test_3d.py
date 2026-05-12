"""3D inference for both ``mode=3d_patch`` and ``mode=3d_full``.

Patch mode: sliding-window inference with overlap blending (mean over votes).
Full mode: single forward per case on the resampled volume, then trilinear
resample of the prediction back to each case's native voxel grid.

Predictions are written as ``<output_dir>/<case>/GED4_pred_3d.nii.gz`` in the
case's native affine either way, so downstream ``scripts/evaluate_predictions.py``
and ``scripts/visualize_prediction.py`` work unchanged for both modes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from liqa_mrgan3d.data.datasets_liqa import (
    LiQA3DFullVolumeDataset,
    LiQA3DPatchDataset,
    load_nifti,
)
from liqa_mrgan3d.models.mrgan3d import MrGANGenerator3D
from liqa_mrgan3d.utils.config import ensure_dir, load_config


def denormalize_from_tanh(volume: np.ndarray) -> np.ndarray:
    return ((volume + 1.0) / 2.0).clip(0.0, 1.0).astype(np.float32)


def _build_generator(config: dict[str, Any], device: torch.device) -> MrGANGenerator3D:
    model = MrGANGenerator3D(
        in_c=int(config.get("input_nc", len(config.get("input_modalities", [])))),
        mid_c=int(config.get("g_base_channels", 32)),
        layers=int(config.get("g_layers", 2)),
        s_layers=int(config.get("g_share_layers", 3)),
    ).to(device)
    return model


def _run_patch_mode(config: dict, device: torch.device, model, output_root: Path) -> None:
    dataset = LiQA3DPatchDataset(config, split="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    accumulators: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"sum": None, "count": None, "sample_dir": None}
    )
    target_modality = str(config.get("target_modality", "GED4"))

    with torch.no_grad():
        for batch in tqdm(loader, desc="3d patch inference"):
            real_a = batch["A"].to(device)
            pred = model(real_a).cpu().numpy()[0, 0]  # [D, H, W]
            sample_id = batch["sample_id"][0]
            sample_dir = batch["sample_dir"][0]
            origin = batch["patch_origin"][0].numpy().astype(int)
            z0, y0, x0 = (int(v) for v in origin)
            d, h, w = pred.shape

            if accumulators[sample_id]["sum"] is None:
                ref, _, _ = load_nifti(Path(sample_dir) / f"{target_modality}.nii.gz")
                accumulators[sample_id]["sum"] = np.zeros(ref.shape, dtype=np.float32)
                accumulators[sample_id]["count"] = np.zeros(ref.shape, dtype=np.float32)
                accumulators[sample_id]["sample_dir"] = sample_dir

            volume_sum = accumulators[sample_id]["sum"]
            volume_count = accumulators[sample_id]["count"]
            assert volume_sum is not None and volume_count is not None
            volume_sum[y0 : y0 + h, x0 : x0 + w, z0 : z0 + d] += pred.transpose(1, 2, 0)
            volume_count[y0 : y0 + h, x0 : x0 + w, z0 : z0 + d] += 1.0

    for sample_id, info in accumulators.items():
        sample_dir = Path(info["sample_dir"])
        _, affine, header = load_nifti(sample_dir / f"{target_modality}.nii.gz")
        volume = info["sum"] / np.maximum(info["count"], 1.0)
        if config.get("normalize_to_tanh", True):
            volume = denormalize_from_tanh(volume)
        sample_output = ensure_dir(output_root / sample_id)
        output_path = sample_output / "GED4_pred_3d.nii.gz"
        nib.save(nib.Nifti1Image(volume.astype(np.float32), affine, header), output_path)
        print(f"Saved {output_path}")


def _run_full_mode(config: dict, device: torch.device, model, output_root: Path) -> None:
    dataset = LiQA3DFullVolumeDataset(config, split="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    target_modality = str(config.get("target_modality", "GED4"))

    with torch.no_grad():
        for batch in tqdm(loader, desc="3d full-volume inference"):
            real_a = batch["A"].to(device)  # [1, C, D, H, W] at volume_size
            pred = model(real_a)  # [1, 1, D, H, W] at volume_size
            sample_id = batch["sample_id"][0]
            sample_dir = Path(batch["sample_dir"][0])
            # Native shape recorded at load time: (H, W, D) as stored in NIfTI.
            native_hw_d = tuple(int(v) for v in batch["native_shape"][0].tolist())
            native_d, native_h, native_w = native_hw_d[2], native_hw_d[0], native_hw_d[1]

            # Resample prediction back to the native voxel grid via trilinear.
            pred_native = F.interpolate(
                pred,
                size=(native_d, native_h, native_w),
                mode="trilinear",
                align_corners=False,
            )[0, 0].cpu().numpy()  # [D, H, W]

            # NIfTI stores (H, W, D), so transpose DHW → HWD.
            volume = pred_native.transpose(1, 2, 0)
            if config.get("normalize_to_tanh", True):
                volume = denormalize_from_tanh(volume)

            _, affine, header = load_nifti(sample_dir / f"{target_modality}.nii.gz")
            sample_output = ensure_dir(output_root / sample_id)
            output_path = sample_output / "GED4_pred_3d.nii.gz"
            nib.save(nib.Nifti1Image(volume.astype(np.float32), affine, header), output_path)
            print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 3D inference (patch or full-volume) and save NIfTI predictions."
    )
    parser.add_argument("--config", default="configs/liqa_3d_patch.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/predictions_3d")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(
        "cuda" if config.get("cuda", True) and torch.cuda.is_available() else "cpu"
    )
    mode = str(config.get("mode", "3d_patch"))

    model = _build_generator(config, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("net_g", checkpoint))
    model.eval()

    output_root = ensure_dir(args.output_dir)
    if mode == "3d_full":
        _run_full_mode(config, device, model, output_root)
    else:
        _run_patch_mode(config, device, model, output_root)


if __name__ == "__main__":
    main()
