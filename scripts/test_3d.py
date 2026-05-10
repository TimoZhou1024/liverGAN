from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from liqa_mrgan3d.data.datasets_liqa import LiQA3DPatchDataset, load_nifti
from liqa_mrgan3d.models.pix2pix_3d import Generator3D
from liqa_mrgan3d.utils.config import ensure_dir, load_config


def denormalize_from_tanh(volume: np.ndarray) -> np.ndarray:
    return ((volume + 1.0) / 2.0).clip(0.0, 1.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3D patch inference and blend patches into NIfTI volumes.")
    parser.add_argument("--config", default="configs/liqa_3d_patch.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/predictions_3d")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if config.get("cuda", True) and torch.cuda.is_available() else "cpu")
    dataset = LiQA3DPatchDataset(config, split="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = Generator3D(
        input_nc=int(config.get("input_nc", len(config.get("input_modalities", [])))),
        output_nc=int(config.get("output_nc", 1)),
        base_channels=int(config.get("g_base_channels", 16)),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("net_g", checkpoint))
    model.eval()

    accumulators: dict[str, dict[str, Any]] = defaultdict(lambda: {"sum": None, "count": None, "sample_dir": None})
    with torch.no_grad():
        for batch in tqdm(loader, desc="3d inference"):
            real_a = batch["A"].to(device)
            pred = model(real_a).cpu().numpy()[0, 0]  # [D, H, W]
            sample_id = batch["sample_id"][0]
            sample_dir = batch["sample_dir"][0]
            origin = batch["patch_origin"][0].numpy().astype(int)
            z0, y0, x0 = (int(v) for v in origin)
            d, h, w = pred.shape

            if accumulators[sample_id]["sum"] is None:
                target_modality = str(config.get("target_modality", "GED4"))
                ref, _, _ = load_nifti(Path(sample_dir) / f"{target_modality}.nii.gz")
                accumulators[sample_id]["sum"] = np.zeros(ref.shape, dtype=np.float32)
                accumulators[sample_id]["count"] = np.zeros(ref.shape, dtype=np.float32)
                accumulators[sample_id]["sample_dir"] = sample_dir

            volume_sum = accumulators[sample_id]["sum"]
            volume_count = accumulators[sample_id]["count"]
            assert volume_sum is not None and volume_count is not None
            volume_sum[y0 : y0 + h, x0 : x0 + w, z0 : z0 + d] += pred.transpose(1, 2, 0)
            volume_count[y0 : y0 + h, x0 : x0 + w, z0 : z0 + d] += 1.0

    output_root = ensure_dir(args.output_dir)
    target_modality = str(config.get("target_modality", "GED4"))
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


if __name__ == "__main__":
    main()
