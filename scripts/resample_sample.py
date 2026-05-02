from __future__ import annotations

import argparse
from pathlib import Path

from liqa_mrgan3d.data.resample import resample_nifti_to_reference


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample LiQA source modalities to a reference modality grid.")
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--modalities", nargs="*", default=["T1", "T2", "DWI_800"])
    parser.add_argument("--reference", default="GED4")
    args = parser.parse_args()

    reference_path = args.sample_dir / f"{args.reference}.nii.gz"
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)

    sample_out = args.output_dir / args.sample_dir.name
    sample_out.mkdir(parents=True, exist_ok=True)
    resample_nifti_to_reference(reference_path, reference_path, sample_out / f"{args.reference}.nii.gz", order=1)

    for modality in args.modalities:
        moving_path = args.sample_dir / f"{modality}.nii.gz"
        if not moving_path.exists():
            raise FileNotFoundError(moving_path)
        output_path = sample_out / f"{modality}.nii.gz"
        resample_nifti_to_reference(moving_path, reference_path, output_path, order=1)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
