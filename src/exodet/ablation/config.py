"""Ablation study configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from exodet.benchmarking.config import _experiment_from_raw, _load_raw_config
from exodet.config.schema import ExperimentConfig
from exodet.exceptions import ConfigurationError

__all__ = [
    "AblationStageConfig",
    "DEFAULT_ABLATION_VARIANTS",
    "load_ablation_stage_config",
]

DEFAULT_ABLATION_VARIANTS: tuple[tuple[str, str, str], ...] = (
    ("cnn_only", "cnn_only", "CNN only"),
    ("transformer_only", "transformer_only", "Transformer only"),
    ("physics_only", "physics_only", "Physics only"),
    ("cnn_transformer", "cnn_transformer", "CNN + Transformer"),
    ("cnn_physics", "physics_only", "CNN + Physics (physics branch only; see docs)"),
    ("transformer_physics", "physics_only", "Transformer + Physics (physics branch only; see docs)"),
    ("fusion", "fusion", "Full hybrid"),
)


@dataclass(frozen=True, slots=True)
class AblationStageConfig:
    """Settings under YAML ``ablation``."""

    enabled: bool = True
    variants: tuple[tuple[str, str, str], ...] = DEFAULT_ABLATION_VARIANTS
    output_dir: str | None = None
    fast_training: dict[str, Any] = field(default_factory=dict)
    ranking_metric: str = "roc_auc"
    backend: str = "sklearn"
    baseline_model: str = "logistic_regression"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "AblationStageConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ConfigurationError("ablation section must be a mapping.")
        variants_raw = raw.get("variants")
        variants = DEFAULT_ABLATION_VARIANTS
        if variants_raw is not None:
            variants = tuple(
                (str(row["id"]), str(row["architecture"]), str(row.get("label", row["id"])))
                for row in variants_raw
            )
        return cls(
            enabled=bool(raw.get("enabled", True)),
            variants=variants,
            output_dir=str(raw["output_dir"]) if raw.get("output_dir") else None,
            fast_training=dict(raw.get("fast_training", {})),
            ranking_metric=str(raw.get("ranking_metric", "roc_auc")),
            backend=str(raw.get("backend", "sklearn")),
            baseline_model=str(raw.get("baseline_model", "logistic_regression")),
        )


def load_ablation_stage_config(
    path: Path | str,
    overrides: list[str] | None = None,
) -> tuple[ExperimentConfig, AblationStageConfig]:
    raw = _load_raw_config(path, overrides)
    experiment = _experiment_from_raw(raw)
    ablation = AblationStageConfig.from_dict(raw.get("ablation"))
    return experiment, ablation
