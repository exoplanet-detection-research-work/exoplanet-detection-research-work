"""Configuration schema and loading utilities.

The public API of this subpackage is:

* :class:`~exodet.config.schema.ExperimentConfig` and its section
  dataclasses, which define the typed shape of every experiment.
* :func:`~exodet.config.loader.load_config`, which parses YAML files
  (with optional base-config inheritance and dotted-key overrides)
  into an :class:`ExperimentConfig`.
"""

from __future__ import annotations

from exodet.config.loader import load_config, load_experiment_config, load_yaml
from exodet.config.schema import (
    ComponentConfig,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    PathsConfig,
    PreprocessingConfig,
    TrainingConfig,
)

__all__ = [
    "ComponentConfig",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "LoggingConfig",
    "ModelConfig",
    "PathsConfig",
    "PreprocessingConfig",
    "TrainingConfig",
    "load_config",
    "load_experiment_config",
    "load_yaml",
]
