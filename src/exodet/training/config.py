"""Research training configuration parsed from YAML."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from exodet.config.schema import TrainingConfig
from exodet.exceptions import ConfigurationError

__all__ = ["ResearchTrainingConfig", "load_research_config"]


@dataclass(frozen=True, slots=True)
class ResearchTrainingConfig:
    """Research-grade training strategy settings.

    Parsed from ``training.trainer.params.research`` without changing
    the top-level :class:`~exodet.config.schema.TrainingConfig` schema.
    """

    enabled: bool = False
    curriculum: dict[str, Any] = field(default_factory=dict)
    imbalance: dict[str, Any] = field(default_factory=dict)
    augmentation: dict[str, Any] = field(default_factory=dict)
    hard_mining: dict[str, Any] = field(default_factory=dict)
    distillation: dict[str, Any] = field(default_factory=dict)
    pretraining: dict[str, Any] = field(default_factory=dict)
    contrastive: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    monitoring: dict[str, Any] = field(default_factory=dict)
    checkpoint_averaging: dict[str, Any] = field(default_factory=dict)
    scientific_validation: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> ResearchTrainingConfig:
        """Builds config from a YAML mapping."""
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ConfigurationError("training.trainer.params.research must be a mapping.")
        return cls(
            enabled=bool(raw.get("enabled", False)),
            curriculum=dict(raw.get("curriculum", {})),
            imbalance=dict(raw.get("imbalance", {})),
            augmentation=dict(raw.get("augmentation", {})),
            hard_mining=dict(raw.get("hard_mining", {})),
            distillation=dict(raw.get("distillation", {})),
            pretraining=dict(raw.get("pretraining", {})),
            contrastive=dict(raw.get("contrastive", {})),
            calibration=dict(raw.get("calibration", {})),
            evaluation=dict(raw.get("evaluation", {})),
            monitoring=dict(raw.get("monitoring", {})),
            checkpoint_averaging=dict(raw.get("checkpoint_averaging", {})),
            scientific_validation=dict(raw.get("scientific_validation", {})),
        )


def load_research_config(config: TrainingConfig) -> ResearchTrainingConfig:
    """Loads research settings from a training config."""
    raw = config.trainer.params.get("research")
    return ResearchTrainingConfig.from_dict(raw)
