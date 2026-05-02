from __future__ import annotations

import argparse
import json
from pathlib import Path

from liqa_mrgan3d.data.nifti_io import REQUIRED_MODALITIES, inspect_nifti


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a LiQA NIfTI sample directory.")
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--modalities", nargs="*", default=list(REQUIRED_MODALITIES))
    args = parser.parse_args()

    report = {}
    for modality in args.modalities:
        path = args.sample_dir / f"{modality}.nii.gz"
        if not path.exists():
            report[modality] = {"missing": True, "path": str(path)}
            continue
        report[modality] = inspect_nifti(path)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
