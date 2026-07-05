"""Checkpoint averaging: EMA, SWA, best-k (Module 11)."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from exodet.ml.callbacks import CALLBACKS, Callback

if TYPE_CHECKING:  # pragma: no cover
    from exodet.ml.trainer import TrainerState

__all__ = [
    "ExponentialMovingAverage",
    "StochasticWeightAveraging",
    "CheckpointEnsemble",
    "EMACallback",
    "SWACallback",
]

logger = logging.getLogger(__name__)


class ExponentialMovingAverage:
    """Maintains EMA of model parameters."""

    def __init__(self, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, Any] | None = None

    def update(self, state_dict: dict[str, Any]) -> None:
        if self.shadow is None:
            self.shadow = copy.deepcopy(state_dict)
            return
        for key, value in state_dict.items():
            self.shadow[key] = self.decay * self.shadow[key] + (1.0 - self.decay) * value

    def apply_to(self, module: Any) -> None:
        if self.shadow is not None:
            module.load_state_dict(self.shadow)


@dataclass
class StochasticWeightAveraging:
    """Collects weights for SWA over late training epochs."""

    start_epoch: int = 5
    snapshots: list[dict[str, Any]] = field(default_factory=list)

    def maybe_collect(self, epoch: int, state_dict: dict[str, Any]) -> None:
        if epoch >= self.start_epoch:
            self.snapshots.append(copy.deepcopy(state_dict))

    def average(self) -> dict[str, Any] | None:
        if not self.snapshots:
            return None
        import torch

        avg: dict[str, Any] = {}
        for key in self.snapshots[0]:
            stacked = torch.stack([s[key].float() for s in self.snapshots])
            avg[key] = stacked.mean(dim=0)
        return avg


class CheckpointEnsemble:
    """Averages top-k checkpoints from a directory."""

    def __init__(self, checkpoint_dir: Path, top_k: int = 3) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.top_k = top_k

    def load_average(self) -> dict[str, Any] | None:
        import torch

        ranked = sorted(self.checkpoint_dir.glob("epoch_*.pt"))[-self.top_k :]
        if not ranked:
            best = self.checkpoint_dir / "best.pt"
            if best.is_file():
                ranked = [best]
            else:
                return None
        states = [torch.load(p, map_location="cpu", weights_only=False)["model_state"] for p in ranked]
        avg: dict[str, Any] = {}
        for key in states[0]:
            avg[key] = sum(s[key].float() for s in states) / len(states)
        return avg


@CALLBACKS.register("ema")
class EMACallback(Callback):
    """Updates exponential moving average each epoch."""

    def __init__(self, decay: float = 0.999, export_path: str | None = None) -> None:
        self.ema = ExponentialMovingAverage(decay)
        self.export_path = Path(export_path) if export_path else None

    def on_epoch_end(self, state: TrainerState, epoch: int) -> None:
        del epoch
        self.ema.update(state.model_state_dict())
        if self.export_path and state.model_module is not None:
            import torch

            self.export_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": self.ema.shadow}, self.export_path)


@CALLBACKS.register("swa")
class SWACallback(Callback):
    """Collects snapshots for stochastic weight averaging."""

    def __init__(self, start_epoch: int = 5, export_path: str | None = None) -> None:
        self.swa = StochasticWeightAveraging(start_epoch=start_epoch)
        self.export_path = Path(export_path) if export_path else None

    def on_epoch_end(self, state: TrainerState, epoch: int) -> None:
        self.swa.maybe_collect(epoch, state.model_state_dict())

    def on_train_end(self, state: TrainerState) -> None:
        del state
        avg = self.swa.average()
        if avg is not None and self.export_path:
            import torch

            self.export_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": avg}, self.export_path)
            logger.info("Exported SWA weights to %s", self.export_path)
