"""Training monitoring callbacks (Module 10)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from exodet.ml.callbacks import Callback
from exodet.registry import Registry
from exodet.utils.io import write_json

if TYPE_CHECKING:  # pragma: no cover
    from exodet.ml.trainer import TrainerState

__all__ = [
    "RESEARCH_CALLBACKS",
    "ResearchMonitoringCallback",
    "HardExampleMiningCallback",
    "build_research_callbacks",
]

logger = logging.getLogger(__name__)

RESEARCH_CALLBACKS: Registry[type[Callback]] = Registry("research callback")


@RESEARCH_CALLBACKS.register("research_monitor")
class ResearchMonitoringCallback(Callback):
    """Logs gradient norms, confidence, class frequencies, throughput."""

    def __init__(self, output_dir: str | Path = "outputs/monitoring") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._batch_times: list[float] = []
        self._t0: float | None = None
        self._history: list[dict[str, Any]] = []

    def on_epoch_begin(self, state: TrainerState, epoch: int) -> None:
        self._t0 = time.perf_counter()
        self._batch_times = []

    def on_batch_end(self, state: TrainerState, batch_idx: int) -> None:
        del batch_idx

        if state.model_module is not None:
            total_norm = 0.0
            for param in state.model_module.parameters():
                if param.grad is not None:
                    total_norm += float(param.grad.data.norm(2) ** 2)
            state.epoch_metrics["grad_norm"] = total_norm**0.5
        self._batch_times.append(time.perf_counter())

    def on_epoch_end(self, state: TrainerState, epoch: int) -> None:
        if state.val_labels:
            labels = np.array(state.val_labels)
            probs = np.array(state.val_probabilities)
            state.epoch_metrics["mean_confidence"] = float(np.mean(np.abs(probs - 0.5) + 0.5))
            unique, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique, counts, strict=True):
                state.epoch_metrics[f"class_freq_{int(label)}"] = float(count / len(labels))
        if self._t0 is not None:
            elapsed = time.perf_counter() - self._t0
            state.epoch_metrics["epoch_seconds"] = elapsed
            if self._batch_times:
                gaps = np.diff([self._t0, *self._batch_times])
                state.epoch_metrics["batch_time_mean"] = float(np.mean(gaps))
        record = {"epoch": epoch, **state.epoch_metrics}
        self._history.append(record)
        write_json(self._history, self.output_dir / "monitoring.json")


@RESEARCH_CALLBACKS.register("hard_example_mining")
class HardExampleMiningCallback(Callback):
    """Tracks hard examples for oversampling in later epochs."""

    def __init__(self, tracker: Any = None) -> None:
        self.tracker = tracker

    def on_epoch_end(self, state: TrainerState, epoch: int) -> None:
        del epoch
        if self.tracker is None or not state.val_probabilities:
            return
        losses = [
            abs(prob - label)
            for prob, label in zip(state.val_probabilities, state.val_labels, strict=True)
        ]
        self.tracker.update(
            tuple(state.val_sample_ids),
            losses,
            state.val_probabilities,
        )


def build_research_callbacks(
    research_config: dict[str, Any],
    hard_tracker: Any = None,
    output_dir: Path | None = None,
) -> list[Callback]:
    """Builds research monitoring callbacks from YAML."""
    callbacks: list[Callback] = []
    mon = research_config.get("monitoring", {})
    if mon.get("enabled", True):
        callbacks.append(
            ResearchMonitoringCallback(
                output_dir=mon.get("output_dir", output_dir or "outputs/monitoring")
            )
        )
    hm = research_config.get("hard_mining", {})
    if hm.get("enabled", False) and hard_tracker is not None:
        callbacks.append(HardExampleMiningCallback(tracker=hard_tracker))
    for spec in mon.get("extra_callbacks", []):
        name = str(spec.get("name", "")).lower()
        params = dict(spec.get("params", {}))
        cls = RESEARCH_CALLBACKS.get(name)
        callbacks.append(cls(**params))
    return callbacks
