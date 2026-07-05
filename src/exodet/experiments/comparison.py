"""Experiment comparison, leaderboards, and rankings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from exodet.experiments.database import ExperimentDatabase, ExperimentRecord
from exodet.experiments.sweeps import _compute_importance

__all__ = [
    "Leaderboard",
    "ComparisonReport",
    "build_leaderboard",
    "compare_experiments",
]


@dataclass
class Leaderboard:
    """Ranked experiment table for one metric."""

    metric: str
    rows: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "rows": self.rows}


@dataclass
class ComparisonReport:
    """Multi-experiment comparison summary."""

    metric_table: dict[str, dict[str, float]]
    runtime_table: dict[str, float]
    rankings: dict[str, list[str]]
    hyperparameter_importance: dict[str, float]
    leaderboards: list[Leaderboard] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_table": self.metric_table,
            "runtime_table": self.runtime_table,
            "rankings": self.rankings,
            "hyperparameter_importance": self.hyperparameter_importance,
            "leaderboards": [lb.to_dict() for lb in self.leaderboards],
        }


def build_leaderboard(
    records: list[ExperimentRecord],
    metric: str = "roc_auc",
    *,
    ascending: bool = False,
    limit: int = 0,
) -> Leaderboard:
    """Build a ranked leaderboard for one metric."""
    rows: list[dict[str, Any]] = []
    for rec in records:
        if metric not in rec.metrics:
            continue
        rows.append(
            {
                "experiment_id": rec.experiment_id,
                "name": rec.name,
                "metric_value": rec.metrics[metric],
                "runtime_seconds": rec.runtime_seconds,
                "tags": list(rec.tags),
                "status": rec.status,
            }
        )
    rows.sort(key=lambda r: r["metric_value"], reverse=not ascending)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    if limit > 0:
        rows = rows[:limit]
    return Leaderboard(metric=metric, rows=rows)


def compare_experiments(
    database: ExperimentDatabase,
    *,
    tags: tuple[str, ...] = (),
    metrics: tuple[str, ...] = ("roc_auc", "accuracy", "f1"),
    ranking_metric: str = "roc_auc",
) -> ComparisonReport:
    """Compare indexed experiments and produce rankings."""
    records = database.search(tags=tags, status="completed")
    metric_table: dict[str, dict[str, float]] = {}
    runtime_table: dict[str, float] = {}
    for rec in records:
        metric_table[rec.experiment_id] = dict(rec.metrics)
        runtime_table[rec.experiment_id] = rec.runtime_seconds

    rankings: dict[str, list[str]] = {}
    leaderboards: list[Leaderboard] = []
    for metric in metrics:
        lb = build_leaderboard(records, metric)
        leaderboards.append(lb)
        rankings[metric] = [row["experiment_id"] for row in lb.rows]

    from exodet.experiments.sweeps import SweepTrial

    sweep_trials = [
        SweepTrial(
            trial_id=i,
            parameters=rec.metadata.get("parameters", {}),
            metrics=rec.metrics,
            runtime_seconds=rec.runtime_seconds,
        )
        for i, rec in enumerate(records)
        if rec.metadata.get("parameters")
    ]
    importance = _compute_importance(sweep_trials, ranking_metric) if sweep_trials else {}

    return ComparisonReport(
        metric_table=metric_table,
        runtime_table=runtime_table,
        rankings=rankings,
        hyperparameter_importance=importance,
        leaderboards=leaderboards,
    )
