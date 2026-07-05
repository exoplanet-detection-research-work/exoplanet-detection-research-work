"""Configuration for experiment orchestration stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from exodet.benchmarking.config import _load_raw_config
from exodet.config.schema import ExperimentConfig
from exodet.exceptions import ConfigurationError

__all__ = [
    "ExperimentsStageConfig",
    "SweepStageConfig",
    "ArtifactsStageConfig",
    "ReproduceStageConfig",
    "load_experiments_stage_config",
    "_STAGE_KEYS",
]

_STAGE_KEYS = frozenset(
    {
        "benchmark",
        "ablation",
        "sensitivity",
        "hyperparameter",
        "reproducibility",
        "inference",
        "report",
        "catalog",
        "compare",
        "experiments",
        "sweep",
        "artifacts",
        "reproduce",
        "templates",
    }
)


def _experiment_from_raw(raw: dict[str, Any]) -> ExperimentConfig:
    experiment_raw = {k: v for k, v in raw.items() if k not in _STAGE_KEYS}
    return ExperimentConfig.from_dict(experiment_raw)


@dataclass(frozen=True, slots=True)
class ExperimentsStageConfig:
    """Settings under YAML ``experiments``."""

    enabled: bool = True
    database_path: str | None = None
    output_dir: str | None = None
    tags: tuple[str, ...] = ()
    template: str | None = None
    parent_id: str | None = None
    inherit_config: bool = True
    stage: str = "train"
    auto_register: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "ExperimentsStageConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ConfigurationError("experiments section must be a mapping.")
        tags = raw.get("tags", [])
        return cls(
            enabled=bool(raw.get("enabled", True)),
            database_path=str(raw["database_path"]) if raw.get("database_path") else None,
            output_dir=str(raw["output_dir"]) if raw.get("output_dir") else None,
            tags=tuple(str(t) for t in tags),
            template=str(raw["template"]) if raw.get("template") else None,
            parent_id=str(raw["parent_id"]) if raw.get("parent_id") else None,
            inherit_config=bool(raw.get("inherit_config", True)),
            stage=str(raw.get("stage", "train")),
            auto_register=bool(raw.get("auto_register", True)),
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class SweepStageConfig:
    """Settings under YAML ``sweep`` for hyperparameter campaigns."""

    enabled: bool = False
    method: str = "grid"
    parameters: dict[str, list[Any]] = field(default_factory=dict)
    random_samples: int = 20
    ranking_metric: str = "roc_auc"
    max_trials: int = 0
    optuna: dict[str, Any] = field(default_factory=dict)
    pbt: dict[str, Any] = field(default_factory=dict)
    model_name: str = "logistic_regression"
    resume: bool = True

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "SweepStageConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ConfigurationError("sweep section must be a mapping.")
        return cls(
            enabled=bool(raw.get("enabled", False)),
            method=str(raw.get("method", "grid")),
            parameters={str(k): list(v) for k, v in dict(raw.get("parameters", {})).items()},
            random_samples=int(raw.get("random_samples", 20)),
            ranking_metric=str(raw.get("ranking_metric", "roc_auc")),
            max_trials=int(raw.get("max_trials", 0)),
            optuna=dict(raw.get("optuna", {})),
            pbt=dict(raw.get("pbt", {})),
            model_name=str(raw.get("model_name", "logistic_regression")),
            resume=bool(raw.get("resume", True)),
        )


@dataclass(frozen=True, slots=True)
class ArtifactsStageConfig:
    """Artifact organization and cleanup policies."""

    enabled: bool = True
    organize: bool = True
    cleanup: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "ArtifactsStageConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ConfigurationError("artifacts section must be a mapping.")
        return cls(
            enabled=bool(raw.get("enabled", True)),
            organize=bool(raw.get("organize", True)),
            cleanup=dict(raw.get("cleanup", {})),
        )


@dataclass(frozen=True, slots=True)
class ReproduceStageConfig:
    """Reproducibility validation settings."""

    enabled: bool = False
    experiment_ids: tuple[str, ...] = ()
    metric_tolerance: float = 1e-4
    probability_tolerance: float = 1e-5
    issue_certificate: bool = True

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "ReproduceStageConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ConfigurationError("reproduce section must be a mapping.")
        ids = raw.get("experiment_ids", [])
        return cls(
            enabled=bool(raw.get("enabled", False)),
            experiment_ids=tuple(str(i) for i in ids),
            metric_tolerance=float(raw.get("metric_tolerance", 1e-4)),
            probability_tolerance=float(raw.get("probability_tolerance", 1e-5)),
            issue_certificate=bool(raw.get("issue_certificate", True)),
        )


def load_experiments_stage_config(
    path: Path | str,
    overrides: list[str] | None = None,
) -> tuple[
    ExperimentConfig,
    ExperimentsStageConfig,
    SweepStageConfig,
    ArtifactsStageConfig,
    ReproduceStageConfig,
    dict[str, Any],
]:
    """Load experiment config and orchestration stage blocks."""
    raw = _load_raw_config(path, overrides)
    experiment = _experiment_from_raw(raw)
    experiments = ExperimentsStageConfig.from_dict(raw.get("experiments"))
    sweep = SweepStageConfig.from_dict(raw.get("sweep"))
    artifacts = ArtifactsStageConfig.from_dict(raw.get("artifacts"))
    reproduce = ReproduceStageConfig.from_dict(raw.get("reproduce"))
    templates_raw = dict(raw.get("templates", {}))
    return experiment, experiments, sweep, artifacts, reproduce, templates_raw
