"""2D slice inference for ``mode: 2d_slice``.

For each test case:
1. Load all modalities + mask, percentile-normalise + tanh-scale (same path as
   training via :func:`load_case_volumes`).
2. For each axial slice ``z`` where the mask has any positive voxel: build the
   2D input ``[1, C, H, W]`` (resized to ``slice_size``), forward through G,
   resample the prediction back to native ``(H, W)``.
3. For slices without liver: fill with zeros.
4. Stack into the native ``(H, W, D)`` volume, denormalise (tanh → [0, 1]),
   save as NIfTI with the original GED4 affine/header.

Output: ``<output_dir>/<case_id>/GED4_pred_2d.nii.gz``. Compatible with
``scripts/evaluate_predictions.py`` (just point ``--pred-name`` at this file).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from liqa_mrgan3d.data.datasets_liqa import (
    _read_list,
    _resolve_mask_path,
    load_case_volumes,
    load_nifti,
    resize_slice_tensor,
)
from liqa_mrgan3d.models.mrgan2d import MrGANGenerator2D
from liqa_mrgan3d.utils.config import ensure_dir, load_config


def denormalize_from_tanh(volume: np.ndarray) -> np.ndarray:
    return ((volume + 1.0) / 2.0).clip(0.0, 1.0).astype(np.float32)


def _build_generator(config: dict[str, Any], device: torch.device) -> MrGANGenerator2D:
    return MrGANGenerator2D(
        in_c=int(config.get("input_nc", len(config.get("input_modalities", [])))),
        mid_c=int(config.get("g_base_channels", 64)),
        layers=int(config.get("g_layers", 2)),
        s_layers=int(config.get("g_share_layers", 3)),
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2D slice inference (mode=2d_slice).")
    parser.add_argument("--config", default="configs/liqa_2d.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/predictions_2d")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(
        "cuda" if config.get("cuda", True) and torch.cuda.is_available() else "cpu"
    )

    model = _build_generator(config, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("net_g", checkpoint))
    model.eval()

    output_root = ensure_dir(args.output_dir)
    target_modality = str(config.get("target_modality", "GED4"))
    modalities: list[str] = list(config.get("input_modalities", ["T1", "T2", "DWI_800"]))
    slice_size = int(config.get("slice_size", 256))
    target_size = (slice_size, slice_size)
    mask_root = Path(config["mask_root"]) if config.get("mask_root") else None
    dataset_root = Path(config["dataset_root"]) if config.get("dataset_root") else None
    mask_suffix = str(config.get("mask_suffix", "GED4_pred.nii.gz"))
    if mask_root is None:
        raise ValueError("mask_root is required for 2D inference (used to identify liver slices).")

    sample_dirs = _read_list(Path(config["test_txt_path"]))
    print(f"2D inference on {len(sample_dirs)} test cases")

    with torch.no_grad():
        for sample_dir in tqdm(sample_dirs, desc="cases"):
            volumes, meta = load_case_volumes(
                config,
                sample_dir,
                modalities=modalities,
                target_modality=target_modality,
            )

            # Read raw mask for the slice filter (the load_case_volumes mask has
            # already been resampled but the same _mask key holds it).
            mask_path = _resolve_mask_path(
                Path(sample_dir),
                mask_root=mask_root,
                dataset_root=dataset_root,
                mask_suffix=mask_suffix,
            )
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask file: {mask_path}")

            native_h, native_w, depth = meta.shape
            output_volume = np.zeros((native_h, native_w, depth), dtype=np.float32)

            mask_volume_native = volumes["_mask"]  # already resampled to GED4 affine
            num_liver_slices = 0
            for z in range(depth):
                if not (mask_volume_native[:, :, z] > 0).any():
                    # Non-liver slice → leave as zero (per user spec).
                    continue

                num_liver_slices += 1
                source_slices = [volumes[m][:, :, z] for m in modalities]
                source = torch.from_numpy(np.stack(source_slices, axis=0)).float().unsqueeze(0)
                source = source.to(device)
                source_resized = resize_slice_tensor(source[0], target_size).unsqueeze(0)

                pred = model(source_resized)  # [1, 1, slice_size, slice_size]

                # Resample prediction back to native (H, W).
                pred_native = F.interpolate(
                    pred,
                    size=(native_h, native_w),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].cpu().numpy()  # [H, W]

                output_volume[:, :, z] = pred_native

            if config.get("normalize_to_tanh", True):
                # Only denormalise the slices we actually wrote; the rest stay 0.
                # tanh→[0,1] would map a 0 into 0.5 — undesirable for empty slices.
                # So we mask: denormalise everywhere then zero out non-liver slices.
                liver_mask_per_slice = (
                    mask_volume_native > 0
                ).any(axis=(0, 1))  # [D] bool
                denorm = denormalize_from_tanh(output_volume)
                denorm[..., ~liver_mask_per_slice] = 0.0
                output_volume = denorm

            _, affine, header = load_nifti(Path(sample_dir) / f"{target_modality}.nii.gz")
            sample_output = ensure_dir(output_root / meta.sample_id)
            output_path = sample_output / "GED4_pred_2d.nii.gz"
            nib.save(
                nib.Nifti1Image(output_volume.astype(np.float32), affine, header),
                output_path,
            )
            print(
                f"Saved {output_path} ({num_liver_slices}/{depth} liver slices predicted)"
            )


if __name__ == "__main__":
    main()
