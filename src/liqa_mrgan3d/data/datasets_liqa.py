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


def resize_volume_tensor(tensor: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    """Resize a CDHW tensor with trilinear interpolation."""
    return F.interpolate(
        tensor.unsqueeze(0),
        size=size,
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)


def parse_3d_size(value: Any, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError(f"Expected a 3D size [D, H, W], got {value}")
    return tuple(int(v) for v in value)


def _read_list(path: str | Path) -> list[Path]:
    with open(path, encoding="utf-8") as f:
        return [Path(line.strip()) for line in f if line.strip()]


def _resolve_mask_path(
    sample_dir: Path,
    *,
    mask_root: Path,
    dataset_root: Path | None,
    mask_suffix: str,
) -> Path:
    """Resolve the GED4 mask path that mirrors ``sample_dir``.

    If ``dataset_root`` is configured, the mask path is
    ``mask_root / sample_dir.relative_to(dataset_root) / mask_suffix`` (works
    symmetrically for ``LiQA_train\\Vendor_*\\<case>`` and
    ``LiQA_val\\Data\\Vendor_*\\<case>``). Otherwise we fall back to
    ``mask_root / vendor / case / mask_suffix`` for backward compatibility.
    """
    sample_dir = Path(sample_dir)
    if dataset_root is not None:
        rel = sample_dir.resolve().relative_to(Path(dataset_root).resolve())
        return mask_root / rel / mask_suffix
    return mask_root / sample_dir.parent.name / sample_dir.name / mask_suffix


def load_case_volumes(
    config: dict[str, Any],
    sample_dir: Path,
    *,
    modalities: list[str],
    target_modality: str,
) -> tuple[dict[str, np.ndarray], VolumeMeta]:
    """Load, resample, normalise all modalities for one case, plus optional mask.

    Returns ``(volumes_dict, meta)`` where ``volumes_dict`` keys are the
    modality names (and ``"_mask"`` when a mask is configured). Modalities are
    resampled to the target modality's voxel grid, percentile-normalised, and
    (if ``normalize_to_tanh`` is set) rescaled to ``[-1, 1]``.

    Extracted from the original ``LiQA3DPatchDataset._load_sample`` so the
    patch and full-volume datasets share a single preprocessing path and stay
    numerically in lock-step.
    """
    sample_dir = Path(sample_dir)
    mask_root = Path(config["mask_root"]) if config.get("mask_root") else None
    dataset_root = Path(config["dataset_root"]) if config.get("dataset_root") else None
    mask_suffix = str(config.get("mask_suffix", "GED4_pred.nii.gz"))
    normalize_to_tanh = bool(config.get("normalize_to_tanh", True))
    resample_to_target = bool(config.get("resample_to_target", True))
    percentile_lower = float(config.get("percentile_lower", 1.0))
    percentile_upper = float(config.get("percentile_upper", 99.0))

    all_modalities = [*modalities, target_modality]
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

        if modality == target_modality:
            affine = modality_affine
            header = modality_header
            reference_shape = volume.shape

    assert affine is not None and header is not None and reference_shape is not None
    for modality, volume in volumes.items():
        if volume.shape != reference_shape:
            if not resample_to_target:
                raise ValueError(
                    f"Shape mismatch in {sample_dir}: {modality}={volume.shape}, "
                    f"{target_modality}={reference_shape}. Run preprocessing/registration first "
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
        volume = percentile_normalize(volume, lower=percentile_lower, upper=percentile_upper)
        if normalize_to_tanh:
            volume = to_minus_one_to_one(volume)
        volumes[modality] = volume

    if mask_root is not None:
        mask_path = _resolve_mask_path(
            sample_dir,
            mask_root=mask_root,
            dataset_root=dataset_root,
            mask_suffix=mask_suffix,
        )
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask file: {mask_path}")
        mask_volume, mask_affine, _ = load_nifti(mask_path)
        if mask_volume.shape != reference_shape:
            mask_volume = resample_like(
                moving=mask_volume,
                moving_affine=mask_affine,
                reference_shape=reference_shape,
                reference_affine=affine,
                order=0,
            )
        mask_volume = (mask_volume > 0.5).astype(np.float32)
        volumes["_mask"] = mask_volume

    meta = VolumeMeta(
        sample_id=sample_dir.name,
        affine=affine,
        header=header,
        shape=reference_shape,
    )
    return volumes, meta


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


class LiQA3DPatchDataset(Dataset[dict[str, Any]]):
    """Patch-based native 3D LiQA dataset.

    Each item returns source modalities as a 3D patch `A` with shape [C, D, H, W]
    and the target GED4 patch `B` with shape [1, D, H, W]. If ``mask_root`` is
    configured, the matching liver mask is loaded, resampled to GED4 voxel space,
    and returned as `M` with the same spatial shape (binary float values).
    """

    def __init__(self, config: dict[str, Any], split: str = "train") -> None:
        self.config = config
        self.split = split
        self.modalities: list[str] = list(config.get("input_modalities", ["T1", "T2", "DWI_800"]))
        self.target_modality: str = str(config.get("target_modality", "GED4"))
        self.patch_size = parse_3d_size(config.get("patch_size"), (16, 128, 128))
        self.patch_stride = parse_3d_size(config.get("patch_stride"), self.patch_size)
        self.normalize_to_tanh = bool(config.get("normalize_to_tanh", True))
        self.resample_to_target = bool(config.get("resample_to_target", True))
        self.max_patches_per_volume = config.get("max_patches_per_volume")
        self.mask_root: Path | None = (
            Path(config["mask_root"]) if config.get("mask_root") else None
        )
        self.dataset_root: Path | None = (
            Path(config["dataset_root"]) if config.get("dataset_root") else None
        )
        self.mask_suffix: str = str(config.get("mask_suffix", "GED4_pred.nii.gz"))

        list_key = f"{split}_txt_path"
        if list_key not in config:
            raise KeyError(f"Missing config key: {list_key}")
        self.sample_dirs = _read_list(config[list_key])
        if not self.sample_dirs:
            raise ValueError(f"No samples found in {config[list_key]}")

        self._cache: dict[Path, tuple[dict[str, np.ndarray], VolumeMeta]] = {}
        self.index: list[tuple[Path, int, int, int]] = []
        self._build_index()

    def _axis_starts(self, length: int, patch: int, stride: int) -> list[int]:
        if length <= patch:
            return [0]
        starts = list(range(0, length - patch + 1, stride))
        last = length - patch
        if starts[-1] != last:
            starts.append(last)
        return starts

    def _mask_path_for(self, sample_dir: Path) -> Path:
        assert self.mask_root is not None
        return _resolve_mask_path(
            Path(sample_dir),
            mask_root=self.mask_root,
            dataset_root=self.dataset_root,
            mask_suffix=self.mask_suffix,
        )

    def _build_index(self) -> None:
        patch_d, patch_h, patch_w = self.patch_size
        stride_d, stride_h, stride_w = self.patch_stride
        for sample_dir in self.sample_dirs:
            target_path = Path(sample_dir) / f"{self.target_modality}.nii.gz"
            if not target_path.exists():
                raise FileNotFoundError(f"Missing target modality {self.target_modality}: {target_path}")
            _, _, shape = read_nifti_meta(target_path)
            h, w, d = shape
            starts_d = self._axis_starts(d, patch_d, stride_d)
            starts_h = self._axis_starts(h, patch_h, stride_h)
            starts_w = self._axis_starts(w, patch_w, stride_w)
            patch_count = 0
            for z in starts_d:
                for y in starts_h:
                    for x in starts_w:
                        self.index.append((Path(sample_dir), z, y, x))
                        patch_count += 1
                        if self.max_patches_per_volume is not None and patch_count >= int(
                            self.max_patches_per_volume
                        ):
                            break
                    if self.max_patches_per_volume is not None and patch_count >= int(
                        self.max_patches_per_volume
                    ):
                        break
                if self.max_patches_per_volume is not None and patch_count >= int(
                    self.max_patches_per_volume
                ):
                    break
        if not self.index:
            raise ValueError("3D patch dataset index is empty. Check patch settings and split files.")

    def _load_sample(self, sample_dir: Path) -> tuple[dict[str, np.ndarray], VolumeMeta]:
        sample_dir = Path(sample_dir)
        if sample_dir in self._cache:
            return self._cache[sample_dir]
        volumes, meta = load_case_volumes(
            self.config,
            sample_dir,
            modalities=self.modalities,
            target_modality=self.target_modality,
        )
        self._cache[sample_dir] = (volumes, meta)
        return volumes, meta

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_dir, z, y, x = self.index[idx]
        volumes, meta = self._load_sample(sample_dir)
        patch_d, patch_h, patch_w = self.patch_size
        h, w, d = meta.shape

        z1 = min(z + patch_d, d)
        y1 = min(y + patch_h, h)
        x1 = min(x + patch_w, w)
        z0 = max(0, z1 - patch_d)
        y0 = max(0, y1 - patch_h)
        x0 = max(0, x1 - patch_w)

        source_patches = [volumes[modality][y0:y1, x0:x1, z0:z1] for modality in self.modalities]
        target_patch = volumes[self.target_modality][y0:y1, x0:x1, z0:z1]

        source = torch.from_numpy(np.stack(source_patches, axis=0)).float().permute(0, 3, 1, 2)
        target = torch.from_numpy(target_patch[None, ...]).float().permute(0, 3, 1, 2)

        expected_size = self.patch_size
        if tuple(source.shape[1:]) != expected_size:
            source = resize_volume_tensor(source, expected_size)
            target = resize_volume_tensor(target, expected_size)

        item: dict[str, Any] = {
            "A": source,
            "B": target,
            "sample_id": meta.sample_id,
            "patch_origin": torch.tensor([z0, y0, x0], dtype=torch.long),
            "sample_dir": str(sample_dir),
        }

        if "_mask" in volumes:
            mask_patch = volumes["_mask"][y0:y1, x0:x1, z0:z1]
            mask = torch.from_numpy(mask_patch[None, ...]).float().permute(0, 3, 1, 2)
            if tuple(mask.shape[1:]) != expected_size:
                # Use nearest-equivalent: trilinear then threshold.
                mask = (resize_volume_tensor(mask, expected_size) > 0.5).float()
            item["M"] = mask

        return item


class LiQA3DFullVolumeDataset(Dataset[dict[str, Any]]):
    """Whole-volume native 3D LiQA dataset for FSDP full-volume training.

    One item per case. Each case's modalities, target, and mask are resampled
    to a single fixed ``volume_size`` (default ``[32, 384, 384]``) so the
    DataLoader returns stackable tensors and FSDP can rely on a static graph.

    ``MrGANGenerator3D`` halves the spatial dimensions 5 times (2 in
    ``Encoder3D`` + 3 in ``ShareNet3D``), so all three dimensions of
    ``volume_size`` MUST be divisible by 32. The default of ``[32, 384, 384]``
    satisfies this and matches typical LiQA axial resolution fairly closely.

    Shares preprocessing (percentile-norm + tanh + mask resampling) with
    :class:`LiQA3DPatchDataset` via the module-level ``load_case_volumes``
    helper so both modes stay numerically in lock-step.
    """

    def __init__(self, config: dict[str, Any], split: str = "train") -> None:
        self.config = config
        self.split = split
        self.modalities: list[str] = list(
            config.get("input_modalities", ["T1", "T2", "DWI_800"])
        )
        self.target_modality: str = str(config.get("target_modality", "GED4"))
        self.volume_size = parse_3d_size(config.get("volume_size"), (32, 384, 384))
        for axis, size in zip("DHW", self.volume_size):
            if size % 32 != 0:
                raise ValueError(
                    f"volume_size[{axis}]={size} is not divisible by 32 — the generator "
                    "halves each spatial dim 5 times (Encoder+ShareNet) and will error."
                )

        list_key = f"{split}_txt_path"
        if list_key not in config:
            raise KeyError(f"Missing config key: {list_key}")
        self.sample_dirs = _read_list(config[list_key])
        if not self.sample_dirs:
            raise ValueError(f"No samples found in {config[list_key]}")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_dir = self.sample_dirs[idx]
        volumes, meta = load_case_volumes(
            self.config,
            sample_dir,
            modalities=self.modalities,
            target_modality=self.target_modality,
        )

        # Stack modalities → [C, H, W, D] → permute to [C, D, H, W] → resample to volume_size.
        source_arr = np.stack([volumes[m] for m in self.modalities], axis=0)
        target_arr = volumes[self.target_modality][None, ...]
        source = torch.from_numpy(source_arr).float().permute(0, 3, 1, 2)
        target = torch.from_numpy(target_arr).float().permute(0, 3, 1, 2)

        source = resize_volume_tensor(source, self.volume_size)
        target = resize_volume_tensor(target, self.volume_size)

        item: dict[str, Any] = {
            "A": source,
            "B": target,
            "sample_id": meta.sample_id,
            "sample_dir": str(sample_dir),
            # Native shape (H, W, D) so test-time inference can resample predictions
            # back to each case's original voxel grid.
            "native_shape": torch.tensor(meta.shape, dtype=torch.long),
        }

        if "_mask" in volumes:
            mask_arr = volumes["_mask"][None, ...]
            mask = torch.from_numpy(mask_arr).float().permute(0, 3, 1, 2)
            # Trilinear resample then threshold → nearest-equivalent for a binary mask.
            mask = (resize_volume_tensor(mask, self.volume_size) > 0.5).float()
            item["M"] = mask

        return item


class LiQA2DSliceDataset(Dataset[dict[str, Any]]):
    """Pure 2D slice dataset, filtered to slices that contain liver tissue.

    For each case in the split file, the GED4 mask under ``mask_root`` is loaded
    and any axial slice with at least one positive mask voxel is admitted to
    the index. Modalities are loaded lazily and cached per case (same pattern
    as :class:`LiQA3DPatchDataset`), then the requested slice is extracted as
    a 2D tensor of shape ``[C, H, W]`` (source) / ``[1, H, W]`` (target/mask).

    Use this for ``mode: 2d_slice`` training. ``slice_size`` (default 256)
    controls bilinear resize of every slice before being returned, so that
    different vendors with different in-plane resolutions yield uniformly
    shaped tensors. Must be divisible by 32 to match
    :class:`MrGANGenerator2D`'s 5 stride-2 halvings.
    """

    def __init__(self, config: dict[str, Any], split: str = "train") -> None:
        self.config = config
        self.split = split
        self.modalities: list[str] = list(
            config.get("input_modalities", ["T1", "T2", "DWI_800"])
        )
        self.target_modality: str = str(config.get("target_modality", "GED4"))
        self.slice_size = int(config.get("slice_size", 256))
        if self.slice_size % 32 != 0:
            raise ValueError(
                f"slice_size={self.slice_size} must be divisible by 32 (generator halves "
                "spatial dims 5 times)"
            )
        self.mask_root: Path | None = (
            Path(config["mask_root"]) if config.get("mask_root") else None
        )
        self.dataset_root: Path | None = (
            Path(config["dataset_root"]) if config.get("dataset_root") else None
        )
        self.mask_suffix: str = str(config.get("mask_suffix", "GED4_pred.nii.gz"))
        if self.mask_root is None:
            raise ValueError(
                "LiQA2DSliceDataset requires mask_root in config (slice filtering relies on it)."
            )

        list_key = f"{split}_txt_path"
        if list_key not in config:
            raise KeyError(f"Missing config key: {list_key}")
        self.sample_dirs = _read_list(config[list_key])
        if not self.sample_dirs:
            raise ValueError(f"No samples found in {config[list_key]}")

        self._cache: dict[Path, tuple[dict[str, np.ndarray], VolumeMeta]] = {}
        # index entries: (sample_dir, z) — only z's where mask has positive voxels.
        self.index: list[tuple[Path, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        """Index only liver-bearing slices.

        Reads the mask NIfTI for each case (small, single channel) without
        touching the heavy modality volumes; per-slice ``sum > 0`` decides
        whether the slice gets indexed. Mask is then forgotten — modalities and
        a fresh mask are loaded together via ``load_case_volumes`` only when
        ``__getitem__`` actually needs them.
        """
        assert self.mask_root is not None
        kept_total = 0
        skipped_total = 0
        for sample_dir in self.sample_dirs:
            mask_path = _resolve_mask_path(
                sample_dir,
                mask_root=self.mask_root,
                dataset_root=self.dataset_root,
                mask_suffix=self.mask_suffix,
            )
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask file: {mask_path}")
            mask_volume, _, _ = load_nifti(mask_path)
            # NIfTI shape is (H, W, D); per-slice axis is the last one.
            depth = mask_volume.shape[2]
            for z in range(depth):
                if (mask_volume[:, :, z] > 0).any():
                    self.index.append((Path(sample_dir), z))
                    kept_total += 1
                else:
                    skipped_total += 1
        if not self.index:
            raise ValueError("LiQA2DSliceDataset index is empty — no slices pass the mask filter.")
        print(
            f"[LiQA2DSliceDataset/{self.split}] indexed {kept_total} slices "
            f"(skipped {skipped_total} non-liver slices) across {len(self.sample_dirs)} cases."
        )

    def _load_sample(self, sample_dir: Path) -> tuple[dict[str, np.ndarray], VolumeMeta]:
        sample_dir = Path(sample_dir)
        if sample_dir in self._cache:
            return self._cache[sample_dir]
        volumes, meta = load_case_volumes(
            self.config,
            sample_dir,
            modalities=self.modalities,
            target_modality=self.target_modality,
        )
        self._cache[sample_dir] = (volumes, meta)
        return volumes, meta

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_dir, z = self.index[idx]
        volumes, meta = self._load_sample(sample_dir)

        # Modality / target / mask volumes are stored as (H, W, D); axial slice z.
        source_slices = [volumes[m][:, :, z] for m in self.modalities]
        target_slice = volumes[self.target_modality][:, :, z]

        source = torch.from_numpy(np.stack(source_slices, axis=0)).float()  # [C, H, W]
        target = torch.from_numpy(target_slice[None, ...]).float()  # [1, H, W]

        # Resize to uniform (slice_size, slice_size).
        target_size = (self.slice_size, self.slice_size)
        source = resize_slice_tensor(source, target_size)
        target = resize_slice_tensor(target, target_size)

        item: dict[str, Any] = {
            "A": source,
            "B": target,
            "sample_id": meta.sample_id,
            "slice_index": int(z),
            "sample_dir": str(sample_dir),
            # Native (H, W) for inference — slice gets resampled back to it.
            "native_hw": torch.tensor([meta.shape[0], meta.shape[1]], dtype=torch.long),
        }

        if "_mask" in volumes:
            mask_slice = volumes["_mask"][:, :, z]
            mask = torch.from_numpy(mask_slice[None, ...]).float()  # [1, H, W]
            mask = (resize_slice_tensor(mask, target_size) > 0.5).float()
            item["M"] = mask

        return item

