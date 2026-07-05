"""Experiment orchestration for large-scale research campaigns."""

from exodet.experiments.config import (
    ArtifactsStageConfig,
    ExperimentsStageConfig,
    ReproduceStageConfig,
    SweepStageConfig,
    load_experiments_stage_config,
)
from exodet.experiments.database import ExperimentDatabase, ExperimentRecord
from exodet.experiments.manager import ExperimentManager
from exodet.experiments.runner import (
    run_experiment,
    run_experiment_sweep,
    run_leaderboard,
    run_reproduce_experiments,
)
from exodet.experiments.templates import EXPERIMENT_TEMPLATES, apply_template, list_templates

__all__ = [
    "ExperimentManager",
    "ExperimentDatabase",
    "ExperimentRecord",
    "ExperimentsStageConfig",
    "SweepStageConfig",
    "ArtifactsStageConfig",
    "ReproduceStageConfig",
    "load_experiments_stage_config",
    "run_experiment",
    "run_experiment_sweep",
    "run_leaderboard",
    "run_reproduce_experiments",
    "EXPERIMENT_TEMPLATES",
    "apply_template",
    "list_templates",
]
