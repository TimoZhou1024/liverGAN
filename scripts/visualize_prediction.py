from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def normalize(volume: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(volume[np.isfinite(volume)], [1, 99])
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - lo) / (hi - lo), 0, 1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Save center-slice visual comparison for GED4 prediction.")
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--pred-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-name", default="GED4.nii.gz")
    args = parser.parse_args()

    target = nib.load(str(args.sample_dir / args.target_name)).get_fdata(dtype=np.float32)
    pred = nib.load(str(args.pred_path)).get_fdata(dtype=np.float32)
    target_n = normalize(target)
    pred_n = normalize(pred)
    z = target.shape[2] // 2
    err = np.abs(pred_n[:, :, z] - target_n[:, :, z])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(target_n[:, :, z].T, cmap="gray", origin="lower")
    axes[0].set_title("Target GED4")
    axes[1].imshow(pred_n[:, :, z].T, cmap="gray", origin="lower")
    axes[1].set_title("Predicted GED4")
    axes[2].imshow(err.T, cmap="magma", origin="lower", vmin=0, vmax=max(float(err.max()), 1e-6))
    axes[2].set_title("Abs Error")
    fig.suptitle(args.sample_dir.name)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
