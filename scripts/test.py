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

from liqa_mrgan3d.data.datasets_liqa import LiQA25DDataset, load_nifti
from liqa_mrgan3d.models.pix2pix_25d import Generator25D
from liqa_mrgan3d.utils.config import ensure_dir, load_config


def denormalize_from_tanh(volume: np.ndarray) -> np.ndarray:
    return ((volume + 1.0) / 2.0).clip(0.0, 1.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LiQA MrGAN 2.5D inference and rebuild NIfTI volumes.")
    parser.add_argument("--config", default="configs/liqa_25d.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/predictions")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if config.get("cuda", True) and torch.cuda.is_available() else "cpu")
    dataset = LiQA25DDataset(config, split="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    input_nc = int(config.get("input_nc", len(config.get("input_modalities", [])) * config.get("slice_window", 3)))
    model = Generator25D(
        input_nc=input_nc,
        output_nc=int(config.get("output_nc", 1)),
        base_channels=int(config.get("g_base_channels", 64)),
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("net_g", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    predictions: dict[str, dict[str, Any]] = defaultdict(lambda: {"slices": {}, "sample_dir": None})
    with torch.no_grad():
        for batch in tqdm(loader, desc="inference"):
            real_a = batch["A"].to(device)
            pred = model(real_a).cpu().numpy()[0, 0]
            sample_id = batch["sample_id"][0]
            slice_index = int(batch["slice_index"].item())
            sample_dir = batch["sample_dir"][0]
            predictions[sample_id]["slices"][slice_index] = pred
            predictions[sample_id]["sample_dir"] = sample_dir

    output_root = ensure_dir(args.output_dir)
    target_modality = str(config.get("target_modality", "GED4"))
    for sample_id, info in predictions.items():
        sample_dir = Path(info["sample_dir"])
        _, affine, header = load_nifti(sample_dir / f"{target_modality}.nii.gz")
        ordered = [info["slices"][z] for z in sorted(info["slices"])]
        volume = np.stack(ordered, axis=2)
        if config.get("normalize_to_tanh", True):
            volume = denormalize_from_tanh(volume)
        sample_output = ensure_dir(output_root / sample_id)
        nib.save(nib.Nifti1Image(volume.astype(np.float32), affine, header), sample_output / "GED4_pred.nii.gz")
        print(f"Saved {sample_output / 'GED4_pred.nii.gz'}")


if __name__ == "__main__":
    main()
