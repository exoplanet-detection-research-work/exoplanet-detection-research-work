"""Configuration for scientific benchmarking stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from exodet.config.loader import _resolve_defaults, apply_overrides, load_yaml
from exodet.config.schema import ExperimentConfig
from exodet.exceptions import ConfigurationError

__all__ = [
    "BenchmarkStageConfig",
    "SensitivityStageConfig",
    "HyperparameterStageConfig",
    "load_benchmark_stage_config",
    "_BENCHMARK_STAGE_KEYS",
    "_load_raw_config",
    "_experiment_from_raw",
]

_BENCHMARK_STAGE_KEYS = frozenset(
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
    }
)


def _load_raw_config(path: Path | str, overrides: list[str] | None) -> dict[str, Any]:
    config_path = Path(path)
    raw = _resolve_defaults(load_yaml(config_path), config_path.parent)
    return apply_overrides(raw, overrides or [])


def _experiment_from_raw(raw: dict[str, Any]) -> ExperimentConfig:
    experiment_raw = {k: v for k, v in raw.items() if k not in _BENCHMARK_STAGE_KEYS}
    return ExperimentConfig.from_dict(experiment_raw)


@dataclass(frozen=True, slots=True)
class BenchmarkStageConfig:
    """Settings under YAML ``benchmark``."""

    enabled: bool = True
    models: tuple[str, ...] = ("xgboost", "random_forest", "logistic_regression", "mlp")
    include_neural: bool = True
    neural_architecture: str = "fusion"
    output_dir: str | None = None
    splits: tuple[str, ...] = ("test",)
    metrics: tuple[str, ...] = ("accuracy", "roc_auc", "pr_auc", "f1", "calibration_error")
    statistics: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    error_analysis: dict[str, Any] = field(default_factory=dict)
    cross_mission: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, Any] = field(default_factory=dict)
    reports: dict[str, Any] = field(default_factory=dict)
    fast_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "BenchmarkStageConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ConfigurationError("benchmark section must be a mapping.")
        models = raw.get("models", ["xgboost", "random_forest", "logistic_regression", "mlp"])
        splits = raw.get("splits", ["test"])
        metrics = raw.get("metrics", ["accuracy", "roc_auc", "pr_auc", "f1", "calibration_error"])
        return cls(
            enabled=bool(raw.get("enabled", True)),
            models=tuple(str(m) for m in models),
            include_neural=bool(raw.get("include_neural", True)),
            neural_architecture=str(raw.get("neural_architecture", "fusion")),
            output_dir=str(raw["output_dir"]) if raw.get("output_dir") else None,
            splits=tuple(str(s) for s in splits),
            metrics=tuple(str(m) for m in metrics),
            statistics=dict(raw.get("statistics", {})),
            calibration=dict(raw.get("calibration", {})),
            error_analysis=dict(raw.get("error_analysis", {})),
            cross_mission=dict(raw.get("cross_mission", {})),
            figures=dict(raw.get("figures", {})),
            reports=dict(raw.get("reports", {})),
            fast_params=dict(raw.get("fast_params", {})),
        )


@dataclass(frozen=True, slots=True)
class SensitivityStageConfig:
    """Settings under YAML ``sensitivity``."""

    enabled: bool = True
    perturbations: tuple[str, ...] = (
        "gaussian_noise",
        "red_noise",
        "missing_cadence",
        "period_offset",
        "epoch_offset",
        "depth_scale",
        "duration_scale",
        "stellar_variability",
    )
    levels: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.1)
    output_name: str = "sensitivity_report"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "SensitivityStageConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ConfigurationError("sensitivity section must be a mapping.")
        return cls(
            enabled=bool(raw.get("enabled", True)),
            perturbations=tuple(str(p) for p in raw.get("perturbations", cls().perturbations)),
            levels=tuple(float(v) for v in raw.get("levels", cls().levels)),
            output_name=str(raw.get("output_name", "sensitivity_report")),
        )


@dataclass(frozen=True, slots=True)
class HyperparameterStageConfig:
    """Settings under YAML ``hyperparameter`` grid search."""

    enabled: bool = False
    parameters: dict[str, list[Any]] = field(default_factory=dict)
    ranking_metric: str = "roc_auc"
    max_trials: int = 0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "HyperparameterStageConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ConfigurationError("hyperparameter section must be a mapping.")
        return cls(
            enabled=bool(raw.get("enabled", False)),
            parameters={str(k): list(v) for k, v in dict(raw.get("parameters", {})).items()},
            ranking_metric=str(raw.get("ranking_metric", "roc_auc")),
            max_trials=int(raw.get("max_trials", 0)),
        )


def load_benchmark_stage_config(
    path: Path | str,
    overrides: list[str] | None = None,
) -> tuple[ExperimentConfig, BenchmarkStageConfig, SensitivityStageConfig, HyperparameterStageConfig, dict[str, Any]]:
    """Loads experiment config and benchmarking stage blocks."""
    raw = _load_raw_config(path, overrides)
    experiment = _experiment_from_raw(raw)
    benchmark = BenchmarkStageConfig.from_dict(raw.get("benchmark"))
    sensitivity = SensitivityStageConfig.from_dict(raw.get("sensitivity"))
    hyperparameter = HyperparameterStageConfig.from_dict(raw.get("hyperparameter"))
    ablation_raw = dict(raw.get("ablation", {}))
    return experiment, benchmark, sensitivity, hyperparameter, ablation_raw
