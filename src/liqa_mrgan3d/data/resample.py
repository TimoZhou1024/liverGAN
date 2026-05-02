from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform


def resample_like(moving: np.ndarray, moving_affine: np.ndarray, reference_shape: tuple[int, int, int], reference_affine: np.ndarray, order: int = 1) -> np.ndarray:
    """Resample `moving` volume onto the reference voxel grid using affine headers.

    This is a header-based resampling fallback. For deformable or optimized
    multimodal registration, preprocess data with SimpleITK before training.
    """
    transform = np.linalg.inv(moving_affine) @ reference_affine
    matrix = transform[:3, :3]
    offset = transform[:3, 3]
    return affine_transform(
        moving,
        matrix=matrix,
        offset=offset,
        output_shape=reference_shape,
        order=order,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)


def resample_nifti_to_reference(moving_path: str | Path, reference_path: str | Path, output_path: str | Path, order: int = 1) -> None:
    moving_img = nib.load(str(moving_path))
    reference_img = nib.load(str(reference_path))
    moving = moving_img.get_fdata(dtype=np.float32)
    resampled = resample_like(
        moving=moving,
        moving_affine=moving_img.affine,
        reference_shape=tuple(int(v) for v in reference_img.shape[:3]),
        reference_affine=reference_img.affine,
        order=order,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(resampled, reference_img.affine, reference_img.header), str(output_path))
