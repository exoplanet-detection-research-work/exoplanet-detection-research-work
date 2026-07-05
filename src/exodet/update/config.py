"""Configuration for incremental dataset update and training resume."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from exodet.benchmarking.config import _load_raw_config
from exodet.config.schema import ExperimentConfig
from exodet.exceptions import ConfigurationError
from exodet.representation.config import RepresentationConfig
from exodet.tce.config import TCESearchConfig

__all__ = [
    "UpdateStageConfig",
    "load_update_stage_config",
    "_UPDATE_STAGE_KEYS",
]

_UPDATE_STAGE_KEYS = frozenset(
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
        "update",
    }
)

_EXPERIMENTS_STAGE_KEYS = frozenset(
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
    experiment_raw = {
        k: v for k, v in raw.items() if k in ExperimentConfig._KEYS
    }
    return ExperimentConfig.from_dict(experiment_raw)


def _tce_from_raw(raw: dict[str, Any]) -> TCESearchConfig:
    tce_raw = {k: v for k, v in raw.items() if k in TCESearchConfig._KEYS}
    return TCESearchConfig.from_dict(tce_raw)


def _representation_from_raw(raw: dict[str, Any]) -> RepresentationConfig:
    rep_raw = {k: v for k, v in raw.items() if k in RepresentationConfig._KEYS}
    return RepresentationConfig.from_dict(rep_raw)


@dataclass(frozen=True, slots=True)
class UpdateStageConfig:
    """Settings under YAML ``update``."""

    enabled: bool = True
    input_tic_ids: tuple[str, ...] = ()
    input_file: str | None = None
    fits_dir: str | None = None
    processed_dir: str | None = None
    missions: tuple[str, ...] = ("TESS",)
    download: dict[str, Any] = field(default_factory=dict)
    parallel_workers: int = 4
    force_reprocess: bool = False
    registry_path: str | None = None
    append_split: str = "train"
    dataset_version: str | None = None
    checkpoint: dict[str, Any] = field(default_factory=dict)
    resume_strategy: str = "continuation"
    parent_experiment_id: str | None = None
    experiment_mode: str = "child"
    evaluation: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    catalog: dict[str, Any] = field(default_factory=dict)
    benchmark: dict[str, Any] = field(default_factory=dict)
    leaderboard: dict[str, Any] = field(default_factory=dict)
    state_dir: str | None = None
    resume_training: bool = True

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> UpdateStageConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ConfigurationError("update section must be a mapping.")
        tic_ids = raw.get("tic_ids", [])
        missions = raw.get("missions", ["TESS"])
        return cls(
            enabled=bool(raw.get("enabled", True)),
            input_tic_ids=tuple(str(t) for t in tic_ids),
            input_file=str(raw["input_file"]) if raw.get("input_file") else None,
            fits_dir=str(raw["fits_dir"]) if raw.get("fits_dir") else None,
            processed_dir=str(raw["processed_dir"]) if raw.get("processed_dir") else None,
            missions=tuple(str(m) for m in missions),
            download=dict(raw.get("download", {})),
            parallel_workers=int(raw.get("parallel_workers", 4)),
            force_reprocess=bool(raw.get("force_reprocess", False)),
            registry_path=str(raw["registry_path"]) if raw.get("registry_path") else None,
            append_split=str(raw.get("append_split", "train")),
            dataset_version=str(raw["dataset_version"]) if raw.get("dataset_version") else None,
            checkpoint=dict(raw.get("checkpoint", {})),
            resume_strategy=str(raw.get("resume_strategy", "continuation")),
            parent_experiment_id=(
                str(raw["parent_experiment_id"]) if raw.get("parent_experiment_id") else None
            ),
            experiment_mode=str(raw.get("experiment_mode", "child")),
            evaluation=dict(raw.get("evaluation", {})),
            report=dict(raw.get("report", {})),
            catalog=dict(raw.get("catalog", {})),
            benchmark=dict(raw.get("benchmark", {})),
            leaderboard=dict(raw.get("leaderboard", {})),
            state_dir=str(raw["state_dir"]) if raw.get("state_dir") else None,
            resume_training=bool(raw.get("resume_training", True)),
        )


def load_update_stage_config(
    path: Path | str,
    overrides: list[str] | None = None,
) -> tuple[
    ExperimentConfig,
    UpdateStageConfig,
    TCESearchConfig,
    RepresentationConfig,
    dict[str, Any],
]:
    """Load experiment, update, TCE, and representation configs from one YAML."""
    raw = _load_raw_config(path, overrides)
    experiment = _experiment_from_raw(raw)
    update = UpdateStageConfig.from_dict(raw.get("update"))
    tce = _tce_from_raw(raw)
    representation = _representation_from_raw(raw)
    return experiment, update, tce, representation, raw
