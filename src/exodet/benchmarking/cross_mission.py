"""Cross-mission evaluation (Kepler, K2, TESS)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from exodet.benchmarking.error_analysis import _infer_mission
from exodet.config.schema import ComponentConfig
from exodet.ml.metrics import compute_all_metrics
from exodet.representation.containers import RepresentationDataset

__all__ = ["CrossMissionReport", "evaluate_cross_mission"]


@dataclass
class CrossMissionReport:
    """Per-mission performance summary."""

    missions: dict[str, dict[str, float]]
    counts: dict[str, int]
    preprocessing_notes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "missions": self.missions,
            "counts": self.counts,
            "preprocessing_notes": self.preprocessing_notes,
        }


_PREPROCESSING_NOTES = {
    "kepler": "Long-cadence Kepler: BKJD time system, quarter stitching, KIC identifiers.",
    "k2": "K2 campaign data: EPIC identifiers, higher pointing jitter; same detrending stack.",
    "tess": "TESS 2-minute cadence: BTJD time system, sector-based splits, TIC identifiers.",
    "unknown": "Mission inferred from target_id prefix when metadata missing.",
}


def evaluate_cross_mission(
    dataset: RepresentationDataset,
    labels: npt.NDArray[np.int_],
    probabilities: npt.NDArray[np.float64],
    *,
    threshold: float = 0.5,
    metric_names: tuple[str, ...] = ("accuracy", "roc_auc", "pr_auc", "f1"),
) -> CrossMissionReport:
    """Evaluate metrics separately for each photometric mission."""
    labels = np.asarray(labels, dtype=np.int_)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    mission_indices: dict[str, list[int]] = {}
    labeled = [i for i, s in enumerate(dataset.samples) if s.label >= 0]
    for offset, idx in enumerate(labeled):
        if offset >= len(labels):
            break
        sample = dataset.samples[idx]
        mission = _infer_mission(sample.target_id, sample.meta)
        mission_indices.setdefault(mission, []).append(offset)

    missions: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    specs = tuple(
        ComponentConfig.from_dict({"name": name, "params": {}}, f"metric.{name}")
        for name in metric_names
    )
    for mission, indices in sorted(mission_indices.items()):
        idx = np.asarray(indices, dtype=np.int_)
        sub_labels = labels[idx]
        sub_probs = probabilities[idx]
        metrics, _ = compute_all_metrics(specs, sub_labels, sub_probs, threshold)
        missions[mission] = metrics
        counts[mission] = int(len(idx))

    notes = {m: _PREPROCESSING_NOTES.get(m, _PREPROCESSING_NOTES["unknown"]) for m in missions}
    return CrossMissionReport(missions=missions, counts=counts, preprocessing_notes=notes)
