"""Scientific error analysis: false positive/negative stratification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from exodet.representation.containers import RepresentationDataset
from exodet.training.curriculum import sample_snr
from exodet.utils.io import ensure_dir, write_json

__all__ = ["ErrorAnalysisReport", "analyze_errors"]


def _infer_mission(target_id: str, meta: dict[str, Any]) -> str:
    mission = str(meta.get("mission", "")).lower()
    if mission in {"kepler", "k2", "tess"}:
        return mission
    tid = target_id.upper()
    if tid.startswith("KIC") or tid.startswith("KOI"):
        return "kepler"
    if tid.startswith("EPIC") or tid.startswith("K2"):
        return "k2"
    if tid.startswith("TIC") or tid.startswith("TOI"):
        return "tess"
    return "unknown"


@dataclass
class ErrorAnalysisReport:
    """Categorized false positives and false negatives."""

    summary: dict[str, int]
    by_period: dict[str, dict[str, float]]
    by_depth: dict[str, dict[str, float]]
    by_duration: dict[str, dict[str, float]]
    by_snr: dict[str, dict[str, float]]
    by_sector: dict[str, dict[str, float]]
    by_magnitude: dict[str, dict[str, float]]
    by_transits: dict[str, dict[str, float]]
    figure_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "by_period": self.by_period,
            "by_depth": self.by_depth,
            "by_duration": self.by_duration,
            "by_snr": self.by_snr,
            "by_sector": self.by_sector,
            "by_magnitude": self.by_magnitude,
            "by_transits": self.by_transits,
            "figure_paths": self.figure_paths,
        }


def _bin_value(value: float, edges: list[float], labels: list[str]) -> str:
    for low, high, label in zip(edges[:-1], edges[1:], labels, strict=True):
        if low <= value < high:
            return label
    return labels[-1]


def _accumulate_stratum(
    store: dict[str, dict[str, float]],
    key: str,
    category: str,
) -> None:
    bucket = store.setdefault(key, {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0, "n": 0.0})
    bucket[category] += 1.0
    bucket["n"] += 1.0


def analyze_errors(
    dataset: RepresentationDataset,
    labels: npt.NDArray[np.int_],
    predictions: npt.NDArray[np.int_],
    probabilities: npt.NDArray[np.float64],
    figure_dir: Path,
    *,
    model_name: str = "model",
) -> ErrorAnalysisReport:
    """Categorize FP/FN by physical and observational strata."""
    import matplotlib.pyplot as plt

    from exodet.visualization.style import apply_publication_style, save_figure

    apply_publication_style()
    ensure_dir(figure_dir)
    labels = np.asarray(labels, dtype=np.int_)
    predictions = np.asarray(predictions, dtype=np.int_)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    summary = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    by_period: dict[str, dict[str, float]] = {}
    by_depth: dict[str, dict[str, float]] = {}
    by_duration: dict[str, dict[str, float]] = {}
    by_snr: dict[str, dict[str, float]] = {}
    by_sector: dict[str, dict[str, float]] = {}
    by_magnitude: dict[str, dict[str, float]] = {}
    by_transits: dict[str, dict[str, float]] = {}

    period_edges = [0.5, 2.0, 10.0, 50.0, 1e6]
    period_labels = ["0.5-2d", "2-10d", "10-50d", ">50d"]
    depth_edges = [0.0, 0.002, 0.01, 0.05, 1.0]
    depth_labels = ["<0.2%", "0.2-1%", "1-5%", ">5%"]
    duration_edges = [0.0, 0.05, 0.15, 0.5, 10.0]
    duration_labels = ["<0.05d", "0.05-0.15d", "0.15-0.5d", ">0.5d"]
    snr_edges = [0.0, 3.0, 7.0, 15.0, 1e6]
    snr_labels = ["<3", "3-7", "7-15", ">15"]

    labeled_indices = [i for i, sample in enumerate(dataset.samples) if sample.label >= 0]
    for offset, idx in enumerate(labeled_indices):
        if offset >= len(labels):
            break
        sample = dataset.samples[idx]
        y = int(labels[offset])
        pred = int(predictions[offset])
        if y == 1 and pred == 1:
            cat = "tp"
        elif y == 0 and pred == 0:
            cat = "tn"
        elif y == 0 and pred == 1:
            cat = "fp"
        else:
            cat = "fn"
        summary[cat] += 1

        cand = sample.candidate
        period_bin = _bin_value(float(cand.period_days), period_edges, period_labels)
        depth_bin = _bin_value(float(cand.depth), depth_edges, depth_labels)
        duration_bin = _bin_value(float(cand.duration_days), duration_edges, duration_labels)
        snr = float(sample_snr(sample))
        snr_bin = _bin_value(snr, snr_edges, snr_labels)
        sector = str(sample.meta.get("sector", sample.meta.get("mission_sector", "unknown")))
        mag = float(sample.meta.get("magnitude", sample.meta.get("kepmag", np.nan)))
        mag_bin = "unknown" if np.isnan(mag) else _bin_value(mag, [0, 10, 13, 16, 30], ["<10", "10-13", "13-16", ">16"])
        n_transits = float(sample.meta.get("n_transits", sample.meta.get("observed_transits", 1)))
        transit_bin = _bin_value(n_transits, [0, 2, 5, 20, 1e6], ["1", "2-4", "5-19", ">=20"])

        _accumulate_stratum(by_period, period_bin, cat)
        _accumulate_stratum(by_depth, depth_bin, cat)
        _accumulate_stratum(by_duration, duration_bin, cat)
        _accumulate_stratum(by_snr, snr_bin, cat)
        _accumulate_stratum(by_sector, sector, cat)
        _accumulate_stratum(by_magnitude, mag_bin, cat)
        _accumulate_stratum(by_transits, transit_bin, cat)

    figure_paths: list[str] = []
    for name, data in (
        ("period", by_period),
        ("depth", by_depth),
        ("snr", by_snr),
    ):
        if not data:
            continue
        keys = sorted(data)
        fp_rates = [
            data[k]["fp"] / max(data[k]["fp"] + data[k]["tn"], 1.0) for k in keys
        ]
        fn_rates = [
            data[k]["fn"] / max(data[k]["fn"] + data[k]["tp"], 1.0) for k in keys
        ]
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(keys))
        ax.bar(x - 0.2, fp_rates, width=0.4, label="FP rate")
        ax.bar(x + 0.2, fn_rates, width=0.4, label="FN rate")
        ax.set_xticks(x, keys, rotation=30, ha="right")
        ax.set_ylabel("Rate")
        ax.set_title(f"Error rates by {name} — {model_name}")
        ax.legend()
        figure_paths.extend(str(p) for p in save_figure(fig, figure_dir, f"{model_name}_errors_{name}"))
        plt.close(fig)

    report = ErrorAnalysisReport(
        summary=summary,
        by_period=by_period,
        by_depth=by_depth,
        by_duration=by_duration,
        by_snr=by_snr,
        by_sector=by_sector,
        by_magnitude=by_magnitude,
        by_transits=by_transits,
        figure_paths=figure_paths,
    )
    write_json(report.to_dict(), figure_dir / f"{model_name}_error_analysis.json")
    return report
