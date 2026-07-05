"""Typed configuration for the scientific inference stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from exodet.config.loader import _resolve_defaults, apply_overrides, load_yaml
from exodet.config.schema import ExperimentConfig
from exodet.exceptions import ConfigurationError

__all__ = [
    "InferenceStageConfig",
    "ReportStageConfig",
    "CatalogStageConfig",
    "load_inference_stage_config",
    "load_report_stage_config",
]

_STAGE_KEYS = frozenset({"inference", "report", "catalog", "compare"})


@dataclass(frozen=True, slots=True)
class InferenceStageConfig:
    """Scientific inference settings from YAML ``inference`` block."""

    enabled: bool = True
    device: str = "auto"
    amp: str = "none"
    batch_size: int = 64
    use_views: str = "both"
    checkpoint_path: str | None = None
    input_dataset: str = "test"
    input_pattern: str = "*.npz"
    streaming: bool = False
    parameter_fit: dict[str, Any] = field(default_factory=dict)
    physical: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    explainability: dict[str, Any] = field(default_factory=dict)
    false_positive: dict[str, Any] = field(default_factory=dict)
    benchmark: dict[str, Any] = field(default_factory=dict)
    parallel_workers: int = 0
    output_name: str = "inference_results"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "InferenceStageConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ConfigurationError("inference section must be a mapping.")
        return cls(
            enabled=bool(raw.get("enabled", True)),
            device=str(raw.get("device", "auto")),
            amp=str(raw.get("amp", "none")),
            batch_size=int(raw.get("batch_size", 64)),
            use_views=str(raw.get("use_views", "both")),
            checkpoint_path=(
                str(raw["checkpoint_path"]) if raw.get("checkpoint_path") else None
            ),
            input_dataset=str(raw.get("input_dataset", "test")),
            input_pattern=str(raw.get("input_pattern", "*.npz")),
            streaming=bool(raw.get("streaming", False)),
            parameter_fit=dict(raw.get("parameter_fit", {})),
            physical=dict(raw.get("physical", {})),
            uncertainty=dict(raw.get("uncertainty", {})),
            explainability=dict(raw.get("explainability", {})),
            false_positive=dict(raw.get("false_positive", {})),
            benchmark=dict(raw.get("benchmark", {})),
            parallel_workers=int(raw.get("parallel_workers", 0)),
            output_name=str(raw.get("output_name", "inference_results")),
        )


@dataclass(frozen=True, slots=True)
class ReportStageConfig:
    """Report generation settings from YAML ``report`` block."""

    enabled: bool = True
    output_dir: str | None = None
    formats: tuple[str, ...] = ("json", "pdf", "csv")
    include_explainability: bool = True
    include_transit_fit: bool = True
    include_uncertainty: bool = True
    top_n: int = 0
    probability_threshold: float = 0.0
    figure_dpi: int = 150

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "ReportStageConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ConfigurationError("report section must be a mapping.")
        formats = raw.get("formats", ["json", "pdf", "csv"])
        return cls(
            enabled=bool(raw.get("enabled", True)),
            output_dir=str(raw["output_dir"]) if raw.get("output_dir") else None,
            formats=tuple(str(f) for f in formats),
            include_explainability=bool(raw.get("include_explainability", True)),
            include_transit_fit=bool(raw.get("include_transit_fit", True)),
            include_uncertainty=bool(raw.get("include_uncertainty", True)),
            top_n=int(raw.get("top_n", 0)),
            probability_threshold=float(raw.get("probability_threshold", 0.0)),
            figure_dpi=int(raw.get("figure_dpi", 150)),
        )


@dataclass(frozen=True, slots=True)
class CatalogStageConfig:
    """Catalog builder settings from YAML ``catalog`` block."""

    enabled: bool = True
    output_name: str = "exoplanet_catalog"
    formats: tuple[str, ...] = ("csv", "json", "parquet")
    sort_by: str = "confidence"
    descending: bool = True
    min_confidence: float = 0.0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "CatalogStageConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ConfigurationError("catalog section must be a mapping.")
        formats = raw.get("formats", ["csv", "json", "parquet"])
        return cls(
            enabled=bool(raw.get("enabled", True)),
            output_name=str(raw.get("output_name", "exoplanet_catalog")),
            formats=tuple(str(f) for f in formats),
            sort_by=str(raw.get("sort_by", "confidence")),
            descending=bool(raw.get("descending", True)),
            min_confidence=float(raw.get("min_confidence", 0.0)),
        )


def _load_raw_config(path: Path | str, overrides: list[str] | None) -> dict[str, Any]:
    config_path = Path(path)
    raw = _resolve_defaults(load_yaml(config_path), config_path.parent)
    return apply_overrides(raw, overrides or [])


def load_inference_stage_config(
    path: Path | str,
    overrides: list[str] | None = None,
) -> tuple[ExperimentConfig, InferenceStageConfig]:
    """Loads experiment config plus inference stage settings."""
    raw = _load_raw_config(path, overrides)
    inference = InferenceStageConfig.from_dict(raw.get("inference"))
    experiment_raw = {k: v for k, v in raw.items() if k not in _STAGE_KEYS}
    experiment = ExperimentConfig.from_dict(experiment_raw)
    return experiment, inference


def load_report_stage_config(
    path: Path | str,
    overrides: list[str] | None = None,
) -> tuple[ExperimentConfig, ReportStageConfig, CatalogStageConfig]:
    """Loads experiment config plus report and catalog settings."""
    raw = _load_raw_config(path, overrides)
    report = ReportStageConfig.from_dict(raw.get("report"))
    catalog = CatalogStageConfig.from_dict(raw.get("catalog"))
    experiment_raw = {k: v for k, v in raw.items() if k not in _STAGE_KEYS}
    experiment = ExperimentConfig.from_dict(experiment_raw)
    return experiment, report, catalog
