"""Hyperparameter sweep orchestration."""

from __future__ import annotations

import itertools
import json
import logging
import random
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from exodet.benchmarking.evaluation import evaluate_probabilities, train_sklearn_baseline
from exodet.config.schema import ComponentConfig, ExperimentConfig
from exodet.experiments.config import SweepStageConfig
from exodet.experiments.manager import ExperimentManager
from exodet.experiments.profiling import ProfileContext
from exodet.ml.trainer import build_trainer
from exodet.representation.containers import RepresentationDataset
from exodet.utils.io import ensure_dir, write_json

__all__ = ["SweepTrial", "SweepResult", "run_sweep"]

logger = logging.getLogger(__name__)


@dataclass
class SweepTrial:
    """One sweep trial outcome."""

    trial_id: int
    parameters: dict[str, Any]
    metrics: dict[str, float]
    runtime_seconds: float
    experiment_id: str | None = None
    rank: int | None = None
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "runtime_seconds": self.runtime_seconds,
            "experiment_id": self.experiment_id,
            "rank": self.rank,
            "status": self.status,
        }


@dataclass
class SweepResult:
    """Complete sweep campaign result."""

    method: str
    ranking_metric: str
    trials: list[SweepTrial] = field(default_factory=list)
    hyperparameter_importance: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "ranking_metric": self.ranking_metric,
            "trials": [t.to_dict() for t in self.trials],
            "hyperparameter_importance": self.hyperparameter_importance,
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


def _iter_random(
    parameters: dict[str, list[Any]],
    n_samples: int,
    seed: int,
) -> Iterator[dict[str, Any]]:
    rng = random.Random(seed)
    keys = sorted(parameters)
    for _ in range(n_samples):
        yield {k: rng.choice(parameters[k]) for k in keys}


def _iter_optuna(
    parameters: dict[str, list[Any]],
    n_trials: int,
    seed: int,
    optuna_cfg: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    try:
        import optuna
    except ImportError:
        logger.warning("optuna not installed; falling back to random search.")
        yield from _iter_random(parameters, n_trials, seed)
        return

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        params: dict[str, Any] = {}
        for key, values in parameters.items():
            if all(isinstance(v, (int, float)) for v in values):
                lo, hi = min(values), max(values)
                params[key] = float(rng.uniform(lo, hi))
            else:
                params[key] = values[int(rng.integers(0, len(values)))]
        yield params


def _compute_importance(trials: list[SweepTrial], metric: str) -> dict[str, float]:
    """Rank hyperparameters by correlation with the ranking metric."""
    if len(trials) < 3:
        return {}
    keys: set[str] = set()
    for t in trials:
        keys.update(t.parameters)
    importance: dict[str, float] = {}
    scores = np.array([t.metrics.get(metric, float("nan")) for t in trials], dtype=np.float64)
    if np.all(np.isnan(scores)):
        return {}
    for key in sorted(keys):
        values = []
        for t in trials:
            val = t.parameters.get(key)
            values.append(float(val) if isinstance(val, (int, float)) else hash(str(val)) % 1000)
        arr = np.asarray(values, dtype=np.float64)
        if np.std(arr) < 1e-12:
            continue
        corr = float(np.corrcoef(arr, scores)[0, 1])
        importance[key] = abs(corr) if not np.isnan(corr) else 0.0
    total = sum(importance.values()) or 1.0
    return {k: v / total for k, v in importance.items()}


def _sklearn_trainer(experiment: ExperimentConfig) -> Any:
    training = replace(
        experiment.training,
        trainer=ComponentConfig(
            name=experiment.training.trainer.name,
            params={**experiment.training.trainer.params, "backend": "sklearn"},
        ),
    )
    return build_trainer(training)


def _load_splits(experiment: ExperimentConfig) -> dict[str, RepresentationDataset]:
    from exodet.ml.runner import _load_splits

    return _load_splits(experiment)


def _run_single_trial(
    experiment: ExperimentConfig,
    trainer: Any,
    splits: dict[str, RepresentationDataset],
    params: dict[str, Any],
    *,
    model_name: str,
    trial_dir: Path,
    threshold: float,
) -> tuple[dict[str, float], float]:
    import exodet.benchmarking.baselines  # noqa: F401
    import exodet.ml.models  # noqa: F401

    start = time.perf_counter()
    model, _, _ = train_sklearn_baseline(
        model_name,
        splits["train"],
        splits["validation"] if len(splits["validation"]) else None,
        trainer,
        trial_dir,
        **params,
    )
    test = splits["test"] if len(splits["test"]) else splits["train"]
    metrics, _, _, _ = evaluate_probabilities(
        model, test, trainer, threshold=threshold,
        metric_names=("accuracy", "roc_auc", "pr_auc", "f1"),
    )
    return metrics, time.perf_counter() - start


def run_sweep(
    experiment: ExperimentConfig,
    sweep: SweepStageConfig,
    *,
    manager: ExperimentManager,
    sweep_id: str,
    seed: int = 0,
) -> SweepResult:
    """Execute a hyperparameter sweep campaign."""
    import exodet.benchmarking.baselines  # noqa: F401
    import exodet.ml.models  # noqa: F401

    if not sweep.enabled:
        raise ValueError("sweep.enabled is false.")
    if not sweep.parameters:
        raise ValueError("sweep.parameters must be non-empty.")

    splits = _load_splits(experiment)
    trainer = _sklearn_trainer(experiment)
    threshold = experiment.evaluation.decision_threshold
    sweep_root = Path(manager.campaign_root) / "sweeps" / sweep_id
    ensure_dir(sweep_root)
    state_path = sweep_root / "sweep_state.json"
    completed_ids: set[int] = set()
    if sweep.resume and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        completed_ids = set(int(i) for i in state.get("completed_trials", []))

    if sweep.method == "grid":
        param_iter = list(_iter_grid(sweep.parameters, sweep.max_trials))
    elif sweep.method == "random":
        n = sweep.max_trials or sweep.random_samples
        param_iter = list(_iter_random(sweep.parameters, n, seed))
    elif sweep.method == "optuna":
        n = sweep.max_trials or sweep.random_samples
        param_iter = list(_iter_optuna(sweep.parameters, n, seed, sweep.optuna))
    elif sweep.method == "pbt":
        param_iter = list(_iter_random(sweep.parameters, sweep.pbt.get("population_size", 8), seed))
    else:
        raise ValueError(f"Unknown sweep method: {sweep.method}")

    result = SweepResult(method=sweep.method, ranking_metric=sweep.ranking_metric)
    for trial_id, params in enumerate(param_iter):
        if trial_id in completed_ids:
            continue
        trial_dir = sweep_root / f"trial_{trial_id:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        try:
            with ProfileContext(output_dir=trial_dir):
                metrics, runtime = _run_single_trial(
                    experiment, trainer, splits, params,
                    model_name=sweep.model_name,
                    trial_dir=trial_dir,
                    threshold=threshold,
                )
            rec = manager.register(name=f"{experiment.experiment_name}_trial_{trial_id:04d}")
            manager.mark_completed(
                rec.experiment_id,
                metrics=metrics,
                artifacts={"checkpoint": str(trial_dir)},
                runtime_seconds=runtime,
            )
            result.trials.append(
                SweepTrial(
                    trial_id=trial_id,
                    parameters=params,
                    metrics=metrics,
                    runtime_seconds=runtime,
                    experiment_id=rec.experiment_id,
                )
            )
            completed_ids.add(trial_id)
            write_json({"completed_trials": sorted(completed_ids)}, state_path)
        except Exception as exc:
            logger.error("Sweep trial %d failed: %s", trial_id, exc)
            result.trials.append(
                SweepTrial(
                    trial_id=trial_id,
                    parameters=params,
                    metrics={},
                    runtime_seconds=0.0,
                    status="failed",
                )
            )

    result.trials.sort(
        key=lambda t: t.metrics.get(sweep.ranking_metric, float("-inf")),
        reverse=True,
    )
    for rank, trial in enumerate(result.trials, start=1):
        trial.rank = rank
    result.hyperparameter_importance = _compute_importance(
        [t for t in result.trials if t.status == "completed"],
        sweep.ranking_metric,
    )
    result.save(sweep_root / "sweep_result.json")
    return result
