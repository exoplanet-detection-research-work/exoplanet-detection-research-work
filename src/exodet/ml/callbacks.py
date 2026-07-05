"""Training callback architecture (Module 9)."""

from __future__ import annotations

import csv
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from exodet.registry import Registry

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from exodet.ml.checkpoints import CheckpointManager
    from exodet.ml.trainer import TrainerState

__all__ = [
    "CALLBACKS",
    "Callback",
    "CallbackList",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "LearningRateMonitorCallback",
    "GradientClippingCallback",
    "PredictionExportCallback",
    "build_callbacks",
]

logger = logging.getLogger(__name__)

CALLBACKS: Registry[type["Callback"]] = Registry("callback")


class Callback(ABC):
    """Hook points invoked by :class:`~exodet.ml.trainer.SupervisedTrainer`."""

    def on_train_begin(self, state: "TrainerState") -> None:
        """Called once before the first training epoch."""

    def on_train_end(self, state: "TrainerState") -> None:
        """Called once after training completes."""

    def on_epoch_begin(self, state: "TrainerState", epoch: int) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(self, state: "TrainerState", epoch: int) -> None:
        """Called at the end of each epoch."""

    def on_batch_begin(self, state: "TrainerState", batch_idx: int) -> None:
        """Called before each training batch."""

    def on_batch_end(self, state: "TrainerState", batch_idx: int) -> None:
        """Called after each training batch."""


class CallbackList:
    """Runs a sequence of callbacks."""

    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        self.callbacks = list(callbacks or [])

    def append(self, callback: Callback) -> None:
        """Adds a callback to the list."""
        self.callbacks.append(callback)

    def on_train_begin(self, state: "TrainerState") -> None:
        for cb in self.callbacks:
            cb.on_train_begin(state)

    def on_train_end(self, state: "TrainerState") -> None:
        for cb in self.callbacks:
            cb.on_train_end(state)

    def on_epoch_begin(self, state: "TrainerState", epoch: int) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(state, epoch)

    def on_epoch_end(self, state: "TrainerState", epoch: int) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(state, epoch)

    def on_batch_begin(self, state: "TrainerState", batch_idx: int) -> None:
        for cb in self.callbacks:
            cb.on_batch_begin(state, batch_idx)

    def on_batch_end(self, state: "TrainerState", batch_idx: int) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(state, batch_idx)


@CALLBACKS.register("early_stopping")
class EarlyStoppingCallback(Callback):
    """Stops training when the monitored metric stops improving."""

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 10,
        min_delta: float = 0.0,
    ) -> None:
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self._best: float | None = None
        self._wait = 0

    def _improved(self, value: float) -> bool:
        if self._best is None:
            return True
        if self.mode == "min":
            return value < self._best - self.min_delta
        return value > self._best + self.min_delta

    def on_epoch_end(self, state: "TrainerState", epoch: int) -> None:
        del epoch
        metrics = state.epoch_metrics
        if self.monitor not in metrics:
            return
        value = metrics[self.monitor]
        if self._improved(value):
            self._best = value
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                state.stop_training = True
                logger.info(
                    "Early stopping after %d epochs without %s improvement.",
                    self.patience,
                    self.monitor,
                )


@CALLBACKS.register("checkpoint")
class CheckpointCallback(Callback):
    """Delegates checkpointing to :class:`~exodet.ml.checkpoints.CheckpointManager`."""

    def __init__(self, manager: "CheckpointManager | None" = None) -> None:
        self.manager = manager

    def on_epoch_end(self, state: "TrainerState", epoch: int) -> None:
        if self.manager is None:
            return
        record = self.manager.save(
            epoch=epoch,
            model_state=state.model_state_dict(),
            optimizer_state=state.optimizer_state_dict(),
            scheduler_state=state.scheduler_state_dict(),
            scaler_state=state.scaler_state_dict(),
            config_snapshot=state.config_snapshot,
            metrics=state.epoch_metrics,
        )
        if record is not None and record.is_best:
            state.best_checkpoint = record.path


@CALLBACKS.register("lr_monitor")
class LearningRateMonitorCallback(Callback):
    """Logs the current learning rate each epoch."""

    def on_epoch_end(self, state: "TrainerState", epoch: int) -> None:
        del epoch
        if state.optimizer is None:
            return
        lrs = [group["lr"] for group in state.optimizer.param_groups]
        state.epoch_metrics["learning_rate"] = float(lrs[0])
        logger.debug("Learning rate: %s", lrs)


@CALLBACKS.register("grad_clip")
class GradientClippingCallback(Callback):
    """Clips gradients by global norm after backward."""

    def __init__(self, max_norm: float = 1.0) -> None:
        self.max_norm = max_norm

    def on_batch_end(self, state: "TrainerState", batch_idx: int) -> None:
        del batch_idx
        if state.model_module is None or self.max_norm <= 0:
            return
        import torch

        torch.nn.utils.clip_grad_norm_(
            state.model_module.parameters(), self.max_norm
        )


@CALLBACKS.register("predict_export")
class PredictionExportCallback(Callback):
    """Exports validation predictions at the end of training."""

    def __init__(self, output_dir: str | Path = "predictions") -> None:
        self.output_dir = Path(output_dir)

    def on_train_end(self, state: "TrainerState") -> None:
        if not state.val_probabilities:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "val_predictions.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "target_id", "label", "probability"])
            for sid, tid, label, prob in zip(
                state.val_sample_ids,
                state.val_target_ids,
                state.val_labels,
                state.val_probabilities,
                strict=True,
            ):
                writer.writerow([sid, tid, label, prob])
        logger.info("Exported validation predictions to %s", path)


@dataclass
class _ConfigurableCallback(Callback):
    """Wraps a built-in callback constructed from YAML params."""

    inner: Callback = field(repr=False)

    def on_train_begin(self, state: "TrainerState") -> None:
        self.inner.on_train_begin(state)

    def on_train_end(self, state: "TrainerState") -> None:
        self.inner.on_train_end(state)

    def on_epoch_begin(self, state: "TrainerState", epoch: int) -> None:
        self.inner.on_epoch_begin(state, epoch)

    def on_epoch_end(self, state: "TrainerState", epoch: int) -> None:
        self.inner.on_epoch_end(state, epoch)

    def on_batch_begin(self, state: "TrainerState", batch_idx: int) -> None:
        self.inner.on_batch_begin(state, batch_idx)

    def on_batch_end(self, state: "TrainerState", batch_idx: int) -> None:
        self.inner.on_batch_end(state, batch_idx)


def build_callbacks(
    specs: tuple[Any, ...],
    checkpoint_manager: "CheckpointManager | None" = None,
    grad_clip_norm: float = 0.0,
) -> CallbackList:
    """Builds callbacks from YAML component configs.

    Args:
        specs: Tuple of :class:`~exodet.config.schema.ComponentConfig`.
        checkpoint_manager: Shared checkpoint manager for checkpoint callback.
        grad_clip_norm: When > 0, appends gradient clipping.

    Returns:
        A :class:`CallbackList` ready for the trainer.
    """
    callbacks: list[Callback] = []
    for spec in specs:
        name = spec.name.lower()
        params = dict(spec.params)
        if name == "checkpoint":
            callbacks.append(CheckpointCallback(manager=checkpoint_manager))
        elif name == "grad_clip":
            callbacks.append(GradientClippingCallback(max_norm=float(
                params.get("max_norm", grad_clip_norm)
            )))
        else:
            cls = CALLBACKS.get(name)
            callbacks.append(cls(**params))
    if grad_clip_norm > 0 and not any(
        isinstance(c, GradientClippingCallback) for c in callbacks
    ):
        callbacks.append(GradientClippingCallback(max_norm=grad_clip_norm))
    return CallbackList(callbacks)
