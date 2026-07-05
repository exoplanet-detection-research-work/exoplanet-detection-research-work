#!/usr/bin/env python3
"""Research-grade evaluation with stratified metrics and figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exodet.config import load_config
from exodet.ml.inference import InferenceEngine
from exodet.ml.trainer import build_trainer
from exodet.models.base import MODELS
from exodet.representation.containers import RepresentationDataset
from exodet.training.config import load_research_config
from exodet.training.evaluation import ResearchEvaluator, ScientificValidator
import exodet.models.registry  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="Research evaluation.")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    args = parser.parse_args()
    config = load_config(args.config)
    research = load_research_config(config.training)
    dataset = RepresentationDataset.load(
        Path(config.paths.processed_dir) / "dataset" / f"{args.split}.npz"
    )
    model = MODELS.build(
        config.model.architecture.name,
        **config.model.architecture.params,
    )
    ckpt = Path(config.paths.checkpoint_dir) / config.experiment_name
    engine = InferenceEngine.from_checkpoint(ckpt, model, trainer=build_trainer(config.training))
    result = engine.predict_batch(dataset)
    arrays = dataset.to_numpy()
    labels = arrays["labels"].astype(np.int_)
    mask = labels >= 0
    labels = labels[mask]
    probs = result.probabilities[mask]

    evaluator = ResearchEvaluator(Path(config.paths.figure_dir))
    report = evaluator.evaluate(
        config.experiment_name,
        args.split,
        labels,
        probs,
        config.evaluation.metrics,
        config.evaluation.decision_threshold,
    )
    validator = ScientificValidator(research.scientific_validation)
    strata = validator.summarize(dataset, labels, probs, config.evaluation.decision_threshold)
    report.strata = strata
    out = Path(config.paths.report_dir) / f"{config.experiment_name}_{args.split}_research.json"
    report.save(out)
    validator.export_table(strata, out.with_suffix(".csv"))
    print(f"Research evaluation saved to {out}")
    print(f"Scores: {report.scores}")


if __name__ == "__main__":
    main()
