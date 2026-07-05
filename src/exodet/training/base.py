"""Abstract training orchestration.

A trainer owns the end-to-end training procedure for a model given a
dataset: batching, epoch loops, checkpointing, and early stopping.
Separating this from :class:`~exodet.models.base.BaseModel` keeps model
definitions declarative and lets training strategies (cross-validation,
curriculum schedules) vary independently of architectures.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from exodet.config.schema import TrainingConfig
from exodet.data.base import BaseDataset
from exodet.models.base import BaseModel
from exodet.registry import Registry

__all__ = ["TrainingResult", "BaseTrainer", "TRAINERS"]

TRAINERS: Registry[BaseTrainer] = Registry("trainer")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Outcome of a completed training run.

    Attributes:
        model: The trained model.
        best_checkpoint: Path to the best saved checkpoint, if any.
        history: Per-epoch metric traces (loss curves, val metrics...).
    """

    model: BaseModel
    best_checkpoint: Path | None = None
    history: dict[str, list[float]] = field(default_factory=dict)


class BaseTrainer(abc.ABC):
    """Abstract training procedure.

    Attributes:
        config: Hyperparameters from the ``training`` config section.
    """

    def __init__(self, config: TrainingConfig) -> None:
        """Initializes the trainer.

        Args:
            config: Training hyperparameters.
        """
        self.config = config

    @abc.abstractmethod
    def train(
        self,
        model: BaseModel,
        train_data: BaseDataset,
        val_data: BaseDataset | None = None,
        checkpoint_dir: Path | None = None,
    ) -> TrainingResult:
        """Runs the full training procedure.

        Args:
            model: The model to train (modified in place and returned
                inside the result).
            train_data: Training samples.
            val_data: Optional validation samples for early stopping.
            checkpoint_dir: Directory for checkpoints; checkpointing is
                skipped when omitted.

        Returns:
            The training result with the trained model and history.

        Raises:
            PipelineError: If training fails irrecoverably.
        """

    def describe(self) -> dict[str, Any]:
        """Summarizes the trainer setup for provenance records.

        Returns:
            A JSON-serializable description of the configuration.
        """
        return {
            "trainer": type(self).__name__,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "early_stopping_patience": self.config.early_stopping_patience,
        }
