from __future__ import annotations

import argparse

from liqa_mrgan3d.trainer.liqa_3d_trainer import LiQA3DTrainer
from liqa_mrgan3d.trainer.liqa_trainer import LiQATrainer
from liqa_mrgan3d.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LiQA MrGAN 2.5D model.")
    parser.add_argument("--config", default="configs/liqa_25d.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    if config.get("mode") == "3d_patch":
        trainer = LiQA3DTrainer(config)
    else:
        trainer = LiQATrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
