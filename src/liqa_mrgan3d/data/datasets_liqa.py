from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from liqa_mrgan3d.data.resample import resample_like


@dataclass(frozen=True)
class VolumeMeta:
    sample_id: str
    affine: np.ndarray
    header: nib.Nifti1Header
    shape: tuple[int, int, int]


def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got shape {data.shape}: {path}")
    return data, image.affine.copy(), image.header.copy()


def read_nifti_meta(path: str | Path) -> tuple[np.ndarray, nib.Nifti1Header, tuple[int, int, int]]:
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got shape {image.shape}: {path}")
    return image.affine.copy(), image.header.copy(), tuple(int(v) for v in image.shape)


def percentile_normalize(
    volume: np.ndarray,
    lower: float = 1.0,
    upper: float = 99.0,
    eps: float = 1e-6,
) -> np.ndarray:
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


def to_minus_one_to_one(volume: np.ndarray) -> np.ndarray:
    return (volume * 2.0 - 1.0).astype(np.float32)


def resize_slice_tensor(tensor: torch.Tensor, size: int | tuple[int, int]) -> torch.Tensor:
    """Resize a CHW tensor with bilinear interpolation."""
    if isinstance(size, int):
        target_size = (size, size)
    else:
        target_size = size
    return F.interpolate(
        tensor.unsqueeze(0),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _read_list(path: str | Path) -> list[Path]:
    with open(path, encoding="utf-8") as f:
        return [Path(line.strip()) for line in f if line.strip()]


class LiQA25DDataset(Dataset[dict[str, Any]]):
    """2.5D LiQA dataset.

    Each item returns neighboring slices from multiple source modalities as `A`
    and the center slice from the target modality as `B`.
    """

    def __init__(self, config: dict[str, Any], split: str = "train") -> None:
        self.config = config
        self.split = split
        self.modalities: list[str] = list(config.get("input_modalities", ["T1", "T2", "DWI_800"]))
        self.target_modality: str = str(config.get("target_modality", "GED4"))
        self.slice_window: int = int(config.get("slice_window", 3))
        if self.slice_window < 1 or self.slice_window % 2 != 1:
            raise ValueError("slice_window must be a positive odd integer")
        self.half_window = self.slice_window // 2
        self.target_size = config.get("target_size", config.get("size", 256))
        self.normalize_to_tanh = bool(config.get("normalize_to_tanh", True))
        self.resample_to_target = bool(config.get("resample_to_target", True))
        self.use_non_empty_target = bool(config.get("use_non_empty_target", False))
        self.min_target_fraction = float(config.get("min_target_fraction", 0.0))
        self.max_slices_per_volume = config.get("max_slices_per_volume")

        list_key = f"{split}_txt_path"
        if list_key not in config:
            raise KeyError(f"Missing config key: {list_key}")
        self.sample_dirs = _read_list(config[list_key])
        if not self.sample_dirs:
            raise ValueError(f"No samples found in {config[list_key]}")

        self._cache: dict[Path, tuple[dict[str, np.ndarray], VolumeMeta]] = {}
        self.index: list[tuple[Path, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        for sample_dir in self.sample_dirs:
            target_path = Path(sample_dir) / f"{self.target_modality}.nii.gz"
            if not target_path.exists():
                raise FileNotFoundError(f"Missing target modality {self.target_modality}: {target_path}")
            _, _, target_shape = read_nifti_meta(target_path)
            depth = target_shape[2]
            slice_indices = range(depth)
            if self.max_slices_per_volume is not None:
                slice_indices = range(min(depth, int(self.max_slices_per_volume)))
            for z in slice_indices:
                self.index.append((sample_dir, z))
        if not self.index:
            raise ValueError("Dataset index is empty. Check filtering and split files.")

    def _load_sample(self, sample_dir: Path) -> tuple[dict[str, np.ndarray], VolumeMeta]:
        sample_dir = Path(sample_dir)
        if sample_dir in self._cache:
            return self._cache[sample_dir]

        all_modalities = [*self.modalities, self.target_modality]
        volumes: dict[str, np.ndarray] = {}
        affines: dict[str, np.ndarray] = {}
        affine: np.ndarray | None = None
        header: nib.Nifti1Header | None = None
        reference_shape: tuple[int, int, int] | None = None

        for modality in all_modalities:
            path = sample_dir / f"{modality}.nii.gz"
            if not path.exists():
                raise FileNotFoundError(f"Missing modality {modality}: {path}")
            volume, modality_affine, modality_header = load_nifti(path)
            volumes[modality] = volume
            affines[modality] = modality_affine

            if modality == self.target_modality:
                affine = modality_affine
                header = modality_header
                reference_shape = volume.shape

        assert affine is not None and header is not None and reference_shape is not None
        for modality, volume in volumes.items():
            if volume.shape != reference_shape:
                if not self.resample_to_target:
                    raise ValueError(
                        f"Shape mismatch in {sample_dir}: {modality}={volume.shape}, "
                        f"{self.target_modality}={reference_shape}. Run preprocessing/registration first "
                        "or set resample_to_target=true."
                    )
                volumes[modality] = resample_like(
                    moving=volume,
                    moving_affine=affines[modality],
                    reference_shape=reference_shape,
                    reference_affine=affine,
                    order=1,
                )

        for modality, volume in list(volumes.items()):
            volume = percentile_normalize(
                volume,
                lower=float(self.config.get("percentile_lower", 1.0)),
                upper=float(self.config.get("percentile_upper", 99.0)),
            )
            if self.normalize_to_tanh:
                volume = to_minus_one_to_one(volume)
            volumes[modality] = volume

        meta = VolumeMeta(
            sample_id=sample_dir.name,
            affine=affine,
            header=header,
            shape=reference_shape,
        )
        self._cache[sample_dir] = (volumes, meta)
        return volumes, meta

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_dir, z = self.index[idx]
        volumes, meta = self._load_sample(sample_dir)
        depth = meta.shape[2]

        source_slices: list[np.ndarray] = []
        for modality in self.modalities:
            volume = volumes[modality]
            for dz in range(-self.half_window, self.half_window + 1):
                zz = min(max(z + dz, 0), depth - 1)
                source_slices.append(volume[:, :, zz])

        target_slice = volumes[self.target_modality][:, :, z]
        if self.use_non_empty_target:
            fraction = float(np.mean(np.abs(target_slice) > 1e-6))
            if fraction < self.min_target_fraction:
                raise ValueError(
                    "use_non_empty_target filtering is only supported during index building when "
                    "precomputed slice metadata is available. Disable it or add a preprocessing index."
                )
        source = torch.from_numpy(np.stack(source_slices, axis=0)).float()
        target = torch.from_numpy(target_slice[None, :, :]).float()

        if self.target_size:
            source = resize_slice_tensor(source, self.target_size)
            target = resize_slice_tensor(target, self.target_size)

        return {
            "A": source,
            "B": target,
            "sample_id": meta.sample_id,
            "slice_index": z,
            "sample_dir": str(sample_dir),
        }
