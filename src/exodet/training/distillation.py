"""Knowledge distillation (Module 5)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from exodet.exceptions import PipelineError
from exodet.ml.data import MlBatch
from exodet.models.base import BaseModel

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["DistillationLoss", "TeacherStudentSetup"]

logger = logging.getLogger(__name__)


@dataclass
class TeacherStudentSetup:
    """Teacher-student distillation configuration.

    Attributes:
        enabled: Whether distillation is active.
        teacher_checkpoint: Path to teacher weights.
        temperature: Softmax temperature for soft labels.
        alpha: Weight on distillation loss vs hard labels.
    """

    enabled: bool = False
    teacher_checkpoint: str | None = None
    temperature: float = 4.0
    alpha: float = 0.5

    def load_teacher(self, student: BaseModel) -> BaseModel:
        """Loads a frozen teacher model from checkpoint."""
        if not self.enabled or not self.teacher_checkpoint:
            raise PipelineError("Distillation enabled but teacher_checkpoint missing.")
        path = Path(self.teacher_checkpoint)
        if not path.is_file():
            raise PipelineError(f"Teacher checkpoint not found: {path}")
        teacher = type(student).load(path)
        if hasattr(teacher, "module"):
            teacher.module.eval()
            for param in teacher.module.parameters():
                param.requires_grad = False
        return teacher


class DistillationLoss:
    """Combines hard-label BCE with soft teacher logits (temperature-scaled)."""

    def __init__(self, base_loss: Any, setup: TeacherStudentSetup, teacher: Any) -> None:
        self.base_loss = base_loss
        self.setup = setup
        self.teacher = teacher

    def __call__(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        batch: MlBatch | None = None,
    ) -> torch.Tensor:
        import torch
        import torch.nn.functional as F

        hard = self.base_loss(student_logits, labels)
        if not self.setup.enabled or self.teacher is None or batch is None:
            return hard

        with torch.no_grad():
            teacher_logits = self.teacher.forward_batch(batch)
        t = self.setup.temperature
        soft_teacher = torch.sigmoid(teacher_logits / t)
        kd = F.binary_cross_entropy_with_logits(
            student_logits / t, soft_teacher, reduction="mean"
        ) * (t * t)
        return self.setup.alpha * kd + (1.0 - self.setup.alpha) * hard
