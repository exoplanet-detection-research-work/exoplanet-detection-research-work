"""Model evaluation helpers for benchmarking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from exodet.config.schema import ComponentConfig
from exodet.ml.metrics import compute_all_metrics, expected_calibration_error
from exodet.ml.trainer import SupervisedTrainer
from exodet.models.base import MODELS, BaseModel
from exodet.representation.containers import RepresentationDataset
from exodet.training.base import TrainingResult

__all__ = [
    "BenchmarkModelResult",
    "evaluate_probabilities",
    "flatten_dataset",
    "train_sklearn_baseline",
]


@dataclass
class BenchmarkModelResult:
    """Training and evaluation outcome for one benchmark model."""

    name: str
    metrics: dict[str, float]
    probabilities: npt.NDArray[np.float64]
    labels: npt.NDArray[np.int_]
    predictions: npt.NDArray[np.int_]
    runtime_seconds: float
    memory_bytes: int | None = None
    checkpoint_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metrics": self.metrics,
            "runtime_seconds": self.runtime_seconds,
            "memory_bytes": self.memory_bytes,
            "checkpoint_path": self.checkpoint_path,
            "extra": self.extra,
        }


def flatten_dataset(
    dataset: RepresentationDataset,
    trainer: SupervisedTrainer,
    *,
    labeled_only: bool = True,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_], npt.NDArray[np.bool_]]:
    """Flatten representation tensors for sklearn evaluation."""
    arrays = dataset.to_numpy()
    mask = arrays["labels"] >= 0 if labeled_only else np.ones(len(arrays["labels"]), dtype=bool)
    features = trainer._flatten_numpy(arrays, mask)
    labels = arrays["labels"][mask].astype(np.int_)
    return features, labels, mask


def train_sklearn_baseline(
    model_name: str,
    train_data: RepresentationDataset,
    val_data: RepresentationDataset | None,
    trainer: SupervisedTrainer,
    checkpoint_dir: Any,
    **model_params: Any,
) -> tuple[BaseModel, TrainingResult, float]:
    """Train a registered sklearn baseline."""
    model = MODELS.build(model_name, **model_params)
    start = time.perf_counter()
    result = trainer.train(model, train_data, val_data, checkpoint_dir=checkpoint_dir)
    runtime = time.perf_counter() - start
    return model, result, runtime


def evaluate_probabilities(
    model: BaseModel,
    dataset: RepresentationDataset,
    trainer: SupervisedTrainer,
    *,
    threshold: float = 0.5,
    metric_names: tuple[str, ...] = (),
) -> tuple[dict[str, float], npt.NDArray[np.float64], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Score a fitted model and compute metrics."""
    features, labels, _ = flatten_dataset(dataset, trainer)
    probabilities = model.predict_proba(features)
    predictions = (probabilities >= threshold).astype(np.int_)
    specs = tuple(
        ComponentConfig.from_dict({"name": name, "params": {}}, f"metric.{name}")
        for name in metric_names
        if name
    )
    metrics, _ = compute_all_metrics(specs, labels, probabilities, threshold)
    metrics["ece"] = expected_calibration_error(labels, probabilities)
    metrics["brier_score"] = float(np.mean((probabilities - labels) ** 2))
    return metrics, probabilities, labels, predictions

