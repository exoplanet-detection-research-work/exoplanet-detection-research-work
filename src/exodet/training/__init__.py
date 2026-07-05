"""Research training strategies extending the ML training ecosystem."""

from __future__ import annotations

from exodet.training.base import TRAINERS, BaseTrainer, TrainingResult
from exodet.training.config import ResearchTrainingConfig, load_research_config

__all__ = [
    "TRAINERS",
    "BaseTrainer",
    "TrainingResult",
    "ResearchTrainingConfig",
    "load_research_config",
]


def __getattr__(name: str) -> object:
    """Lazy exports to avoid import cycles."""
    _lazy = {
        "ResearchSupervisedTrainer": "research_trainer",
        "CurriculumScheduler": "curriculum",
        "ClassImbalanceHandler": "curriculum",
        "TrainingAugmentationPipeline": "augmentation",
        "ResearchDataModule": "data",
        "DistillationLoss": "distillation",
        "MaskedPretrainer": "pretraining",
        "ContrastivePretrainer": "contrastive",
        "TemperatureScaler": "calibration",
        "ResearchEvaluator": "evaluation",
        "ScientificValidator": "evaluation",
        "benchmark_matrix": "benchmarking",
    }
    if name in _lazy:
        import importlib

        mod = importlib.import_module(f"exodet.training.{_lazy[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
