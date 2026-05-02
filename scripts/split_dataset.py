from __future__ import annotations

import argparse
import random
from pathlib import Path

from liqa_mrgan3d.data.nifti_io import REQUIRED_MODALITIES, find_samples, write_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test split files for LiQA data.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("data"), type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--val-ratio", default=0.1, type=float)
    parser.add_argument("--test-ratio", default=0.1, type=float)
    parser.add_argument("--modalities", nargs="*", default=list(REQUIRED_MODALITIES))
    args = parser.parse_args()

    samples = find_samples(args.data_root, tuple(args.modalities))
    if not samples:
        raise SystemExit(f"No complete samples found under {args.data_root}")

    rng = random.Random(args.seed)
    rng.shuffle(samples)

    n_total = len(samples)
    n_test = max(1, int(round(n_total * args.test_ratio))) if n_total > 2 else 0
    n_val = max(1, int(round(n_total * args.val_ratio))) if n_total > 2 else 0
    n_train = max(0, n_total - n_val - n_test)

    train = samples[:n_train]
    val = samples[n_train : n_train + n_val]
    test = samples[n_train + n_val :]

    write_split(train, args.output_dir / "train.txt")
    write_split(val, args.output_dir / "val.txt")
    write_split(test, args.output_dir / "test.txt")

    print(f"Found {n_total} samples")
    print(f"train={len(train)} val={len(val)} test={len(test)}")
    print(f"Wrote split files to {args.output_dir}")


if __name__ == "__main__":
    main()
