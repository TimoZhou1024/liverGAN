"""Build a small vendor-balanced split for the MrGAN-aligned 3D experiment.

Train and validation cases come from ``LiQA_train``; the held-out test cases
come from ``LiQA_val`` so we never train on data that overlaps with the
official validation set. Only cases that have all required modalities and a
matching GED4 liver mask under ``GED4_masks_pred`` are kept.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from liqa_mrgan3d.data.nifti_io import REQUIRED_MODALITIES, has_required_modalities, write_split


def find_balanced_cases(
    cases_root: Path,
    mask_root: Path,
    mask_suffix: str,
    vendors: list[str],
    modalities: tuple[str, ...],
) -> dict[str, list[Path]]:
    """Group cases by vendor, keeping only those with full modalities + mask.

    ``cases_root`` is the directory that *directly* contains ``Vendor_A`` /
    ``Vendor_B1`` / ``Vendor_B2`` subfolders (for LiQA_train this is the root
    itself; for LiQA_val it is the ``Data`` subdirectory). ``mask_root`` is the
    mirror directory under ``GED4_masks_pred`` with the same vendor layout.
    """
    per_vendor: dict[str, list[Path]] = {}
    for vendor in vendors:
        vendor_dir = cases_root / vendor
        if not vendor_dir.exists():
            raise FileNotFoundError(vendor_dir)
        cases: list[Path] = []
        for case_dir in sorted(vendor_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            if not has_required_modalities(case_dir, modalities):
                continue
            mask_path = mask_root / vendor / case_dir.name / mask_suffix
            if not mask_path.exists():
                continue
            cases.append(case_dir)
        per_vendor[vendor] = cases
    return per_vendor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-cases-root",
        type=Path,
        default=Path(r"D:\BaiduNetdiskDownload\LiQA_train"),
        help="Directory containing Vendor_* subfolders for train/val cases.",
    )
    parser.add_argument(
        "--test-cases-root",
        type=Path,
        default=Path(r"D:\BaiduNetdiskDownload\LiQA_val\Data"),
        help="Directory containing Vendor_* subfolders for held-out test cases.",
    )
    parser.add_argument(
        "--train-mask-root",
        type=Path,
        default=Path(r"D:\BaiduNetdiskDownload\GED4_masks_pred\LiQA_train"),
        help="Mirror of --train-cases-root under GED4_masks_pred.",
    )
    parser.add_argument(
        "--test-mask-root",
        type=Path,
        default=Path(r"D:\BaiduNetdiskDownload\GED4_masks_pred\LiQA_val\Data"),
        help="Mirror of --test-cases-root under GED4_masks_pred.",
    )
    parser.add_argument("--mask-suffix", default="GED4_pred.nii.gz")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--vendors", nargs="*", default=["Vendor_A", "Vendor_B1", "Vendor_B2"])
    parser.add_argument("--n-train", type=int, default=20)
    parser.add_argument("--n-val", type=int, default=5)
    parser.add_argument("--n-test", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modalities", nargs="*", default=list(REQUIRED_MODALITIES))
    args = parser.parse_args()

    train_pool = find_balanced_cases(
        args.train_cases_root,
        args.train_mask_root,
        args.mask_suffix,
        args.vendors,
        tuple(args.modalities),
    )
    test_pool = find_balanced_cases(
        args.test_cases_root,
        args.test_mask_root,
        args.mask_suffix,
        args.vendors,
        tuple(args.modalities),
    )

    train_needed = args.n_train + args.n_val
    train_all: list[Path] = []
    val_all: list[Path] = []
    test_all: list[Path] = []

    print("Train/val pools (from LiQA_train):")
    for vendor, cases in train_pool.items():
        print(f"  {vendor}: {len(cases)} available")
        if len(cases) < train_needed:
            raise SystemExit(
                f"Not enough usable cases in train pool {vendor}: "
                f"have {len(cases)}, need {train_needed}."
            )
        rng = random.Random(args.seed + hash(vendor) % 100000)
        rng.shuffle(cases)
        train_all.extend(cases[: args.n_train])
        val_all.extend(cases[args.n_train : train_needed])

    print("Test pool (from LiQA_val):")
    for vendor, cases in test_pool.items():
        print(f"  {vendor}: {len(cases)} available")
        if len(cases) < args.n_test:
            raise SystemExit(
                f"Not enough usable cases in test pool {vendor}: "
                f"have {len(cases)}, need {args.n_test}."
            )
        rng = random.Random(args.seed + hash("test_" + vendor) % 100000)
        rng.shuffle(cases)
        test_all.extend(cases[: args.n_test])

    write_split(train_all, args.output_dir / "train.txt")
    write_split(val_all, args.output_dir / "val.txt")
    write_split(test_all, args.output_dir / "test.txt")

    print(
        f"\nWrote {len(train_all)} train / {len(val_all)} val / {len(test_all)} test cases "
        f"to {args.output_dir}/{{train,val,test}}.txt"
    )


if __name__ == "__main__":
    main()
