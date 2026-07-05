"""Automated hyperparameter grid search for benchmarking."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from exodet.benchmarking.evaluation import BenchmarkModelResult, evaluate_probabilities, train_sklearn_baseline
from exodet.config.schema import ExperimentConfig
from exodet.ml.trainer import SupervisedTrainer
from exodet.representation.containers import RepresentationDataset
from exodet.utils.io import write_json

__all__ = ["HyperparameterTrial", "HyperparameterStudy", "run_hyperparameter_study"]


@dataclass
class HyperparameterTrial:
    """One hyperparameter configuration trial."""

    parameters: dict[str, Any]
    metrics: dict[str, float]
    runtime_seconds: float
    rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": self.parameters,
            "metrics": self.metrics,
            "runtime_seconds": self.runtime_seconds,
            "rank": self.rank,
        }


@dataclass
class HyperparameterStudy:
    """Ranked summary of hyperparameter trials."""

    ranking_metric: str
    trials: list[HyperparameterTrial] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranking_metric": self.ranking_metric,
            "trials": [t.to_dict() for t in self.trials],
        }

    def save(self, path: Path) -> Path:
        return write_json(self.to_dict(), path)


def _iter_grid(parameters: dict[str, list[Any]], max_trials: int) -> Iterator[dict[str, Any]]:
    keys = sorted(parameters)
    values = [parameters[k] for k in keys]
    for index, combo in enumerate(itertools.product(*values)):
        if max_trials > 0 and index >= max_trials:
            break
        yield dict(zip(keys, combo, strict=True))


def run_hyperparameter_study(
    experiment: ExperimentConfig,
    train_data: RepresentationDataset,
    val_data: RepresentationDataset | None,
    test_data: RepresentationDataset,
    trainer: SupervisedTrainer,
    parameters: dict[str, list[Any]],
    *,
    model_name: str = "mlp",
    ranking_metric: str = "roc_auc",
    max_trials: int = 0,
    checkpoint_root: Path,
    threshold: float = 0.5,
) -> HyperparameterStudy:
    """Grid search over trainer/model hyperparameters."""
    study = HyperparameterStudy(ranking_metric=ranking_metric)
    for trial_id, params in enumerate(_iter_grid(parameters, max_trials)):
        ckpt = checkpoint_root / f"trial_{trial_id:04d}"
        ckpt.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        model, _, _ = train_sklearn_baseline(
            model_name, train_data, val_data, trainer, ckpt, **params
        )
        metrics, _, _, _ = evaluate_probabilities(
            model, test_data, trainer, threshold=threshold, metric_names=("accuracy", "roc_auc", "pr_auc", "f1")
        )
        runtime = time.perf_counter() - start
        study.trials.append(
            HyperparameterTrial(parameters=params, metrics=metrics, runtime_seconds=runtime)
        )

    study.trials.sort(
        key=lambda t: t.metrics.get(ranking_metric, float("-inf")),
        reverse=True,
    )
    for rank, trial in enumerate(study.trials, start=1):
        trial.rank = rank
    return study
