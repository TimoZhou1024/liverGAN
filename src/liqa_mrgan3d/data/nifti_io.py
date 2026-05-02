from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

REQUIRED_MODALITIES = ("T1", "T2", "DWI_800", "GED4")


def has_required_modalities(sample_dir: str | Path, modalities: tuple[str, ...] = REQUIRED_MODALITIES) -> bool:
    sample_dir = Path(sample_dir)
    return all((sample_dir / f"{modality}.nii.gz").exists() for modality in modalities)


def find_samples(data_root: str | Path, modalities: tuple[str, ...] = REQUIRED_MODALITIES) -> list[Path]:
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(data_root)
    samples = [path for path in data_root.iterdir() if path.is_dir() and has_required_modalities(path, modalities)]
    return sorted(samples)


def inspect_nifti(path: str | Path) -> dict[str, object]:
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size:
        intensity = {
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "p1": float(np.percentile(finite, 1)),
            "p99": float(np.percentile(finite, 99)),
        }
    else:
        intensity = {"min": 0.0, "max": 0.0, "mean": 0.0, "p1": 0.0, "p99": 0.0}
    return {
        "path": str(path),
        "shape": tuple(int(v) for v in image.shape),
        "spacing": tuple(float(v) for v in image.header.get_zooms()[:3]),
        "affine": image.affine.tolist(),
        "dtype": str(image.get_data_dtype()),
        "intensity": intensity,
    }


def write_split(paths: list[Path], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for path in paths:
            f.write(f"{path}\n")
