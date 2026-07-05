#!/usr/bin/env python3
"""Run masked or contrastive encoder pretraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exodet.config import load_config
from exodet.models.base import MODELS
from exodet.representation.containers import RepresentationDataset
from exodet.training.config import load_research_config
from exodet.training.contrastive import run_contrastive_pretraining
from exodet.training.pretraining import run_masked_pretraining
import exodet.models.registry  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="Encoder pretraining.")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument(
        "--mode",
        choices=["masked", "contrastive"],
        default="masked",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    research = load_research_config(config.training)
    split = RepresentationDataset.load(
        Path(config.paths.processed_dir) / "dataset" / "train.npz"
    )
    model = MODELS.build(
        config.model.architecture.name,
        **config.model.architecture.params,
    )
    out = Path(config.paths.checkpoint_dir) / config.experiment_name / "pretrain"
    if args.mode == "contrastive":
        path = run_contrastive_pretraining(model, split, research.contrastive, out)
    else:
        path = run_masked_pretraining(model, split, research.pretraining, out)
    print(f"Pretraining complete: {path}")


if __name__ == "__main__":
    main()
