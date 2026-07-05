"""Experiment tracking (Module 10)."""

from __future__ import annotations

import csv
import importlib
import logging
import platform
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from exodet.utils.io import ensure_dir, write_json

__all__ = [
    "EnvironmentInfo",
    "collect_environment_info",
    "ExperimentTracker",
    "CsvLogger",
    "TensorBoardLogger",
    "WandbLogger",
]

logger = logging.getLogger(__name__)


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def collect_environment_info() -> "EnvironmentInfo":
    """Collects library versions and git commit for reproducibility.

    Returns:
        Environment metadata snapshot.
    """
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for package in ("numpy", "scipy", "torch", "sklearn", "xgboost"):
        try:
            mod = importlib.import_module(package)
            versions[package] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[package] = "not installed"
    return EnvironmentInfo(
        git_commit=_git_commit_hash(),
        library_versions=versions,
    )


@dataclass(frozen=True, slots=True)
class EnvironmentInfo:
    """Reproducibility metadata.

    Attributes:
        git_commit: Current git HEAD hash, if available.
        library_versions: Installed package versions.
    """

    git_commit: str | None
    library_versions: dict[str, str]


class ExperimentTracker(ABC):
    """Abstract experiment logger."""

    @abstractmethod
    def log_hyperparameters(self, params: Mapping[str, Any]) -> None:
        """Records hyperparameters."""

    @abstractmethod
    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Records scalar metrics at a training step."""

    @abstractmethod
    def close(self) -> None:
        """Flushes and closes the tracker."""


class CsvLogger(ExperimentTracker):
    """Appends per-epoch metrics to a CSV file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self._fieldnames: list[str] | None = None
        self._file = self.path.open("a", encoding="utf-8", newline="")
        self._writer: csv.DictWriter | None = None

    def log_hyperparameters(self, params: Mapping[str, Any]) -> None:
        meta_path = self.path.with_suffix(".meta.json")
        write_json(dict(params), meta_path)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        row = {"step": step, **{k: float(v) for k, v in metrics.items()}}
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            if self.path.stat().st_size == 0:
                self._writer.writeheader()
        assert self._writer is not None
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class TensorBoardLogger(ExperimentTracker):
    """Logs scalars to TensorBoard."""

    def __init__(self, log_dir: Path | str) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "TensorBoard logging requires 'tensorboard' (bundled with torch)."
            ) from exc
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_hyperparameters(self, params: Mapping[str, Any]) -> None:
        text = "\n".join(f"{k}: {v}" for k, v in sorted(params.items()))
        self._writer.add_text("hyperparameters", text, 0)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        for key, value in metrics.items():
            self._writer.add_scalar(key, float(value), step)

    def close(self) -> None:
        self._writer.close()


class WandbLogger(ExperimentTracker):
    """Optional Weights & Biases integration."""

    def __init__(
        self,
        project: str,
        run_name: str | None = None,
        entity: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "W&B tracking requires 'wandb'; install with 'pip install wandb'."
            ) from exc
        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            name=run_name,
            entity=entity,
            config=dict(config or {}),
            reinit=True,
        )

    def log_hyperparameters(self, params: Mapping[str, Any]) -> None:
        self._wandb.config.update(dict(params), allow_val_change=True)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        self._wandb.log({k: float(v) for k, v in metrics.items()}, step=step)

    def close(self) -> None:
        self._wandb.finish()


@dataclass
class MultiTracker(ExperimentTracker):
    """Fans out logging to multiple backends."""

    trackers: list[ExperimentTracker] = field(default_factory=list)

    def log_hyperparameters(self, params: Mapping[str, Any]) -> None:
        for tracker in self.trackers:
            tracker.log_hyperparameters(params)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        for tracker in self.trackers:
            tracker.log_metrics(metrics, step)

    def close(self) -> None:
        for tracker in self.trackers:
            tracker.close()


def build_tracker(
    tracking_config: dict[str, Any],
    output_dir: Path,
    experiment_name: str,
    hyperparameters: Mapping[str, Any] | None = None,
) -> ExperimentTracker:
    """Builds experiment trackers from YAML config.

    Args:
        tracking_config: ``training.trainer.params.tracking`` block.
        output_dir: Run output directory.
        experiment_name: Experiment name for run naming.
        hyperparameters: Hyperparameters to log at start.

    Returns:
        A single tracker or :class:`MultiTracker`.
    """
    enabled = tracking_config.get("enabled", True)
    if not enabled:
        return _NullTracker()

    backends = tracking_config.get("backends", ["csv"])
    if isinstance(backends, str):
        backends = [backends]

    trackers: list[ExperimentTracker] = []
    for backend in backends:
        backend = backend.lower()
        if backend == "csv":
            trackers.append(CsvLogger(output_dir / "metrics.csv"))
        elif backend == "tensorboard":
            trackers.append(TensorBoardLogger(output_dir / "tensorboard"))
        elif backend == "wandb":
            trackers.append(
                WandbLogger(
                    project=tracking_config.get("project", "exodet"),
                    run_name=tracking_config.get("run_name", experiment_name),
                    entity=tracking_config.get("entity"),
                    config=hyperparameters,
                )
            )
        else:
            logger.warning("Unknown tracking backend '%s'; skipping.", backend)

    env = collect_environment_info()
    meta = {
        "experiment_name": experiment_name,
        "git_commit": env.git_commit,
        "library_versions": env.library_versions,
        **dict(hyperparameters or {}),
    }
    write_json(meta, output_dir / "run_metadata.json")

    if not trackers:
        return _NullTracker()
    if len(trackers) == 1:
        tracker = trackers[0]
        tracker.log_hyperparameters(meta)
        return tracker
    multi = MultiTracker(trackers=trackers)
    multi.log_hyperparameters(meta)
    return multi


class _NullTracker(ExperimentTracker):
    def log_hyperparameters(self, params: Mapping[str, Any]) -> None:
        del params

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        del metrics, step

    def close(self) -> None:
        return None
