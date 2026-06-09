"""Evaluate generated GED4 NIfTI predictions with two metric conventions.

**Ours** (default, existing logic, backward-compatible):
    - percentile_normalize(pred, p1, p99) + percentile_normalize(target, p1, p99)
    - MAE / RMSE / PSNR / SSIM with data_range=1.0, SSIM per-slice then mean

**MrGAN** (from ``reference/MrGAN/trainer/p2pTrainer_v6.py:386-428``):
    - min-max normalize each volume independently: ``(x - min) / (max - min)``
    - MAE (global + inside liver mask when mask is provided)
    - PSNR / SSIM with data_range=descending from the min-max range

Pass ``--metric-style both`` (the default) to get both sets side-by-side in the
same CSV. ``--mask-root`` and ``--mask-suffix`` are optional; when provided,
a third column ``mae_masked_mrgan`` is added for MrGAN-style liver-only MAE.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def percentile_normalize(
    volume: np.ndarray,
    lower: float = 1.0,
    upper: float = 99.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Clip to [p_lower, p_upper] then rescale to [0, 1]."""
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


def minmax_normalize(
    volume: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Per-volume min-max rescale to [0, 1].

    Port of ``reference/MrGAN/trainer/p2pTrainer_v6.py:414-428``
    ``P2p_Trainer_v6.normalize()``.
    """
    finite = np.isfinite(volume)
    if not finite.any():
        return np.zeros_like(volume, dtype=np.float32)
    lo = float(np.min(volume[finite]))
    hi = float(np.max(volume[finite]))
    if hi - lo < eps:
        return np.zeros_like(volume, dtype=np.float32)
    vol = volume.copy()
    vol = (vol - lo) / (hi - lo + eps)
    vol[~finite] = 0.0
    return vol.astype(np.float32)


def load_volume(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def _resolve_mask_path(
    sample_dir: Path,
    mask_root: Path,
    mask_suffix: str,
    dataset_root: Path | None = None,
) -> Path:
    """Mirror the dataset-level mask resolver from ``datasets_liqa.py``."""
    sample_dir = Path(sample_dir)
    if dataset_root is not None:
        rel = sample_dir.resolve().relative_to(Path(dataset_root).resolve())
        return mask_root / rel / mask_suffix
    return mask_root / sample_dir.parent.name / sample_dir.name / mask_suffix


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def compute_metrics_ours(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    percentile_lower: float = 1.0,
    percentile_upper: float = 99.0,
) -> dict[str, float]:
    """Existing logic — percentile-normalised, data_range=1.0."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")
    pred = np.nan_to_num(pred.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pred_n = percentile_normalize(pred, percentile_lower, percentile_upper)
    target_n = percentile_normalize(target, percentile_lower, percentile_upper)

    diff = pred_n - target_n
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff * diff))
    rmse = float(np.sqrt(mse))
    psnr = float(peak_signal_noise_ratio(target_n, pred_n, data_range=1.0))
    psnr_nosc = float(peak_signal_noise_ratio(pred_n, target_n, data_range=1.0))  # alias

    ssim_values: list[float] = []
    for z in range(target.shape[2]):
        t = target_n[:, :, z]
        p = pred_n[:, :, z]
        if np.std(t) < 1e-6 or np.std(p) < 1e-6:
            continue
        ssim_values.append(float(structural_similarity(t, p, data_range=1.0)))
    ssim = float(np.mean(ssim_values)) if ssim_values else float("nan")

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "psnr": psnr,
        "ssim": ssim,
    }


def compute_metrics_mrgan(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """MrGAN-style metric: min-max normalise each volume independently.

    Mirrors ``reference/MrGAN/trainer/p2pTrainer_v6.py:386-428``:
    - ``MAE``: mean absolute diff over all foreground voxels (where target != -1
      sentinel in the original). When ``mask`` is provided, a liver-only MAE is
      also reported.
    - ``PSNR`` / ``SSIM``: min-max → [0, 1] then ``data_range=1.0``.

    MrGAN divides the MAE by 2 because its images live in [-1, 1] (tanh). Our
    predictions are already denormalised to [0, 1], so we report the raw
    per-voxel absolute difference without the /2 correction. To compare against
    the paper, divide ``mae_mrgan`` by 2.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")
    pred = np.nan_to_num(pred.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    pred_n = minmax_normalize(pred)
    target_n = minmax_normalize(target)

    mae_global = float(np.mean(np.abs(pred_n - target_n)))

    if mask is not None and mask.any():
        mae_masked = float(np.mean(np.abs(pred_n[mask > 0] - target_n[mask > 0])))
    else:
        mae_masked = float("nan")

    psnr = float(peak_signal_noise_ratio(target_n, pred_n, data_range=1.0))

    ssim_vals: list[float] = []
    for z in range(target.shape[2]):
        t = target_n[:, :, z]
        p = pred_n[:, :, z]
        if np.std(t) < 1e-6 or np.std(p) < 1e-6:
            continue
        ssim_vals.append(float(structural_similarity(t, p, data_range=1.0)))
    ssim = float(np.mean(ssim_vals)) if ssim_vals else float("nan")

    return {
        "mae_mrgan": mae_global,
        "mae_masked_mrgan": mae_masked,
        "psnr_mrgan": psnr,
        "ssim_mrgan": ssim,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generated GED4 NIfTI predictions (ours + MrGAN conventions)."
    )
    parser.add_argument("--pred-root", default="outputs/predictions_3d", type=Path)
    parser.add_argument("--test-list", default="data/test.txt", type=Path)
    parser.add_argument("--pred-name", default="GED4_pred_3d.nii.gz")
    parser.add_argument("--target-name", default="GED4.nii.gz")
    parser.add_argument("--output-csv", default="outputs/eval_3d_metrics.csv", type=Path)

    # Percentile-normalisation params (ours only; MrGAN uses min-max).
    parser.add_argument("--percentile-lower", default=1.0, type=float)
    parser.add_argument("--percentile-upper", default=99.0, type=float)

    # Mask (optional — enables MrGAN liver-only MAE).
    parser.add_argument("--mask-root", default=None, type=Path,
                        help="Root of GED4_masks_pred. When set, per-case mask is loaded "
                             "for MrGAN's masked-MAE.")
    parser.add_argument("--dataset-root", default=None, type=Path,
                        help="Root of raw data (for resolving mask path symmetrically).")
    parser.add_argument("--mask-suffix", default="GED4_pred.nii.gz")

    # Metric style.
    parser.add_argument(
        "--metric-style",
        default="both",
        choices=["ours", "mrgan", "both"],
        help="Which metric convention(s) to compute (default: both).",
    )

    args = parser.parse_args()

    sample_dirs = [
        Path(line.strip())
        for line in args.test_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

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

        # ----- Mask (if requested) -----
        mask: np.ndarray | None = None
        if args.mask_root is not None:
            mask_path = _resolve_mask_path(
                sample_dir,
                mask_root=args.mask_root,
                mask_suffix=args.mask_suffix,
                dataset_root=args.dataset_root,
            )
            if mask_path.exists():
                mask = nib.load(str(mask_path)).get_fdata(dtype=np.float32)
                mask = (mask > 0.5).astype(np.float32)
                if mask.shape != target.shape:
                    print(f"  [warn] mask shape {mask.shape} != target shape {target.shape}, "
                          f"skipping masked-MAE for {sample_id}")
                    mask = None
            else:
                print(f"  [warn] mask not found: {mask_path}, skipping masked-MAE for {sample_id}")

        row: dict[str, str | float] = {"sample_id": sample_id, "pred_path": str(pred_path)}

        # ----- our metrics -----
        if args.metric_style in ("ours", "both"):
            m_ours = compute_metrics_ours(
                pred, target,
                percentile_lower=args.percentile_lower,
                percentile_upper=args.percentile_upper,
            )
            row.update(m_ours)
            print(
                f"{sample_id}: [ours]  "
                f"MAE={m_ours['mae']:.4f}  RMSE={m_ours['rmse']:.4f}  "
                f"PSNR={m_ours['psnr']:.2f}  SSIM={m_ours['ssim']:.4f}"
            )

        # ----- MrGAN metrics -----
        if args.metric_style in ("mrgan", "both"):
            m_mrgan = compute_metrics_mrgan(pred, target, mask=mask)
            row.update(m_mrgan)
            masked_str = f"  masked-MAE={m_mrgan['mae_masked_mrgan']:.4f}" if mask is not None else ""
            print(
                f"{'':<12}[mrgan] "
                f"MAE={m_mrgan['mae_mrgan']:.4f}  "
                f"PSNR={m_mrgan['psnr_mrgan']:.2f}  "
                f"SSIM={m_mrgan['ssim_mrgan']:.4f}"
                f"{masked_str}"
            )

        rows.append(row)

    if not rows:
        raise SystemExit("No predictions found to evaluate.")

    # Build fieldnames from row keys preserving order.
    fieldnames = list(rows[0].keys())

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ----- Summary -----
    print("\nSummary")
    for key in fieldnames:
        if key in ("sample_id", "pred_path"):
            continue
        values = np.array([float(row[key]) for row in rows if key in row], dtype=np.float32)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        print(f"  {key}: mean={float(values.mean()):.4f}  std={float(values.std()):.4f}")
    print(f"Saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
