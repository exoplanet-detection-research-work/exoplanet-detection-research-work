#!/usr/bin/env python3
"""Post-training temperature scaling calibration."""

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
from exodet.training.calibration import fit_temperature_scaling, plot_reliability_diagram
import exodet.models.registry  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate model probabilities.")
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    val = RepresentationDataset.load(
        Path(config.paths.processed_dir) / "dataset" / "validation.npz"
    )
    if len(val) == 0:
        val = RepresentationDataset.load(
            Path(config.paths.processed_dir) / "dataset" / "test.npz"
        )
    model = MODELS.build(
        config.model.architecture.name,
        **config.model.architecture.params,
    )
    ckpt = Path(config.paths.checkpoint_dir) / config.experiment_name
    engine = InferenceEngine.from_checkpoint(ckpt, model, trainer=build_trainer(config.training))
    result = engine.predict_batch(val)
    arrays = val.to_numpy()
    labels = arrays["labels"].astype(np.int_)
    mask = labels >= 0
    labels = labels[mask]
    probs = result.probabilities[mask]
    logits = np.log(probs / np.clip(1 - probs, 1e-6, 1.0))
    scaler = fit_temperature_scaling(logits, labels)
    out = ckpt / "temperature.json"
    scaler.save(out)
    plot_reliability_diagram(
        labels,
        scaler.calibrate(logits),
        Path(config.paths.figure_dir),
        "calibrated_reliability",
    )
    print(f"Saved temperature scaler to {out} (T={scaler.temperature:.3f})")


if __name__ == "__main__":
    main()
