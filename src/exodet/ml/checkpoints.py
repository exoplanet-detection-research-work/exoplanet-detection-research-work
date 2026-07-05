"""Checkpoint manager (Module 7)."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from exodet.utils.io import ensure_dir, write_json

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["CheckpointRecord", "CheckpointManager"]

logger = logging.getLogger(__name__)


def _require_torch():
    import torch

    return torch


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """Metadata for one saved checkpoint.

    Attributes:
        path: Checkpoint file path.
        epoch: Training epoch (1-based).
        metric_name: Monitored metric key.
        metric_value: Monitored metric value.
        is_best: Whether this is the current best checkpoint.
    """

    path: Path
    epoch: int
    metric_name: str
    metric_value: float
    is_best: bool = False


@dataclass
class CheckpointManager:
    """Saves best, last, and top-k validation checkpoints with full state.

    Attributes:
        directory: Checkpoint output directory.
        monitor: Metric key to rank checkpoints (lower is better when
            ``mode='min'``).
        mode: ``min`` or ``max``.
        save_last: Whether to always save the most recent epoch.
        save_best: Whether to track the best validation checkpoint.
        top_k: Number of top checkpoints to retain (``0`` disables).
        filename_template: Template with ``{epoch}`` and ``{metric}`` placeholders.
    """

    directory: Path
    monitor: str = "val_loss"
    mode: str = "min"
    save_last: bool = True
    save_best: bool = True
    top_k: int = 3
    filename_template: str = "epoch_{epoch:04d}_{metric}_{value:.4f}.pt"

    _best_value: float | None = field(default=None, init=False, repr=False)
    _top_records: list[CheckpointRecord] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        ensure_dir(self.directory)
        if self.mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'.")

    def is_improvement(self, value: float) -> bool:
        """Returns whether ``value`` beats the current best."""
        if self._best_value is None:
            return True
        if self.mode == "min":
            return value < self._best_value
        return value > self._best_value

    def save(
        self,
        *,
        epoch: int,
        model_state: dict[str, Any],
        optimizer_state: dict[str, Any] | None = None,
        scheduler_state: dict[str, Any] | None = None,
        scaler_state: dict[str, Any] | None = None,
        config_snapshot: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> CheckpointRecord | None:
        """Writes a checkpoint and updates best/last/top-k tracking.

        Args:
            epoch: Current epoch (1-based).
            model_state: Model ``state_dict``.
            optimizer_state: Optimizer state.
            scheduler_state: Scheduler state.
            scaler_state: AMP scaler state.
            config_snapshot: Frozen config for reproducibility.
            metrics: Epoch metrics (must include ``monitor`` when ranking).
            extra: Additional JSON-serializable metadata.

        Returns:
            The checkpoint record when a ranked checkpoint was saved,
            else ``None`` for last-only saves without ranking.
        """
        torch = _require_torch()
        metrics = metrics or {}
        metric_value = float(metrics.get(self.monitor, float("nan")))
        payload: dict[str, Any] = {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer_state or {},
            "scheduler_state": scheduler_state or {},
            "scaler_state": scaler_state or {},
            "config_snapshot": config_snapshot or {},
            "metrics": metrics,
            "extra": extra or {},
        }

        if self.save_last:
            last_path = self.directory / "last.pt"
            torch.save(payload, last_path)
            write_json(
                {"epoch": epoch, "metrics": metrics, "path": str(last_path)},
                self.directory / "last.json",
            )

        improved = self.is_improvement(metric_value) if self.monitor in metrics else False
        record: CheckpointRecord | None = None

        if self.save_best and improved and self.monitor in metrics:
            self._best_value = metric_value
            best_path = self.directory / "best.pt"
            torch.save(payload, best_path)
            shutil.copy2(best_path, self.directory / f"best_epoch_{epoch:04d}.pt")
            record = CheckpointRecord(
                path=best_path,
                epoch=epoch,
                metric_name=self.monitor,
                metric_value=metric_value,
                is_best=True,
            )
            logger.info(
                "New best checkpoint at epoch %d: %s=%.6f",
                epoch,
                self.monitor,
                metric_value,
            )

        if self.top_k > 0 and self.monitor in metrics:
            fname = self.filename_template.format(
                epoch=epoch, metric=self.monitor, value=metric_value
            )
            ranked_path = self.directory / fname
            torch.save(payload, ranked_path)
            self._top_records.append(
                CheckpointRecord(
                    path=ranked_path,
                    epoch=epoch,
                    metric_name=self.monitor,
                    metric_value=metric_value,
                    is_best=improved,
                )
            )
            self._top_records.sort(
                key=lambda r: r.metric_value,
                reverse=self.mode == "max",
            )
            while len(self._top_records) > self.top_k:
                removed = self._top_records.pop()
                if removed.path.is_file() and removed.path.name not in (
                    "best.pt",
                    "last.pt",
                ):
                    removed.path.unlink(missing_ok=True)
            if record is None:
                record = CheckpointRecord(
                    path=ranked_path,
                    epoch=epoch,
                    metric_name=self.monitor,
                    metric_value=metric_value,
                )

        return record

    def load(
        self, path: Path | None = None, prefer: str = "best"
    ) -> dict[str, Any]:
        """Loads a checkpoint payload for resume.

        Args:
            path: Explicit checkpoint path; when ``None``, uses ``prefer``.
            prefer: ``best``, ``last``, or a filename stem.

        Returns:
            Deserialized checkpoint dict.
        """
        torch = _require_torch()
        if path is None:
            if prefer == "best":
                path = self.directory / "best.pt"
            elif prefer == "last":
                path = self.directory / "last.pt"
            else:
                path = self.directory / prefer
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        logger.info("Loaded checkpoint from %s (epoch %s)", path, payload.get("epoch"))
        return payload

    @property
    def best_checkpoint(self) -> Path | None:
        """Path to the best checkpoint, if saved."""
        path = self.directory / "best.pt"
        return path if path.is_file() else None
