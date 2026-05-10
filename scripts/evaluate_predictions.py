from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def percentile_normalize(volume: np.ndarray, lower: float = 1.0, upper: float = 99.0, eps: float = 1e-6) -> np.ndarray:
    finite = np.isfinite(volume)
    if not finite.any():
        return np.zeros_like(volume, dtype=np.float32)
    values = volume[finite]
    lo, hi = np.percentile(values, [lower, upper])
    if hi - lo < eps:
        return np.zeros_like(volume, dtype=np.float32)
    volume = np.clip(volume, lo, hi)
    volume = (volume - lo) / (hi - lo + eps)
    volume[~finite] = 0.0
    return volume.astype(np.float32)


def load_volume(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")
    pred = np.nan_to_num(pred.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    diff = pred - target
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff * diff))
    rmse = float(np.sqrt(mse))
    psnr = float(peak_signal_noise_ratio(target, pred, data_range=1.0))

    ssim_values: list[float] = []
    for z in range(target.shape[2]):
        t = target[:, :, z]
        p = pred[:, :, z]
        if np.std(t) < 1e-6 and np.std(p) < 1e-6:
            continue
        ssim_values.append(float(structural_similarity(t, p, data_range=1.0)))
    ssim = float(np.mean(ssim_values)) if ssim_values else float("nan")
    return {"mae": mae, "mse": mse, "rmse": rmse, "psnr": psnr, "ssim": ssim}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated GED4 NIfTI predictions.")
    parser.add_argument("--pred-root", default="outputs/predictions_3d", type=Path)
    parser.add_argument("--test-list", default="data/test.txt", type=Path)
    parser.add_argument("--pred-name", default="GED4_pred_3d.nii.gz")
    parser.add_argument("--target-name", default="GED4.nii.gz")
    parser.add_argument("--output-csv", default="outputs/eval_3d_metrics.csv", type=Path)
    parser.add_argument("--percentile-lower", default=1.0, type=float)
    parser.add_argument("--percentile-upper", default=99.0, type=float)
    args = parser.parse_args()

    sample_dirs = [Path(line.strip()) for line in args.test_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, str | float]] = []
    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        pred_path = args.pred_root / sample_id / args.pred_name
        target_path = sample_dir / args.target_name
        if not pred_path.exists():
            print(f"skip missing prediction: {pred_path}")
            continue
        pred = load_volume(pred_path)
        target = load_volume(target_path)
        pred = percentile_normalize(pred, args.percentile_lower, args.percentile_upper)
        target = percentile_normalize(target, args.percentile_lower, args.percentile_upper)
        metrics = compute_metrics(pred, target)
        row: dict[str, str | float] = {"sample_id": sample_id, "pred_path": str(pred_path), **metrics}
        rows.append(row)
        print(
            f"{sample_id}: MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} "
            f"PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f}"
        )

    if not rows:
        raise SystemExit("No predictions found to evaluate.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "mae", "mse", "rmse", "psnr", "ssim", "pred_path"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    numeric_keys = ["mae", "mse", "rmse", "psnr", "ssim"]
    print("\nSummary")
    for key in numeric_keys:
        values = np.array([float(row[key]) for row in rows], dtype=np.float32)
        values = values[np.isfinite(values)]
        print(f"{key}: mean={float(values.mean()):.4f} std={float(values.std()):.4f}")
    print(f"Saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
