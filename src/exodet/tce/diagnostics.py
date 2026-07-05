"""Tabular diagnostic exports of the TCE stage (CSV and JSON).

Complements the figures in :mod:`exodet.visualization.tce` with the
machine-readable products: the ranked candidate table and the run-level
detection summary.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from exodet.tce.candidate import STATUS_CANDIDATE, TransitCandidate
from exodet.tce.pipeline import TCEResult
from exodet.utils.io import ensure_dir, write_json

__all__ = ["export_candidates_csv", "write_detection_summary"]

logger = logging.getLogger(__name__)

_CSV_FIELDS = (
    "rank",
    "candidate_id",
    "target_id",
    "status",
    "period_days",
    "epoch_days",
    "duration_days",
    "depth",
    "depth_err",
    "n_transits",
    "n_expected_transits",
    "snr",
    "sde",
    "power",
    "fap",
    "sectors",
    "quality_flags",
    "rejection_reason",
)


def export_candidates_csv(
    candidates: list[TransitCandidate], path: Path | str
) -> Path:
    """Writes the full candidate table (all statuses) as CSV.

    Args:
        candidates: Candidates to export, accepted and rejected.
        path: Destination CSV file.

    Returns:
        The written file path.
    """
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "rank": candidate.meta.get("rank", ""),
                    "candidate_id": candidate.candidate_id,
                    "target_id": candidate.target_id,
                    "status": candidate.status,
                    "period_days": candidate.period_days,
                    "epoch_days": candidate.epoch_days,
                    "duration_days": candidate.duration_days,
                    "depth": candidate.depth,
                    "depth_err": candidate.depth_err,
                    "n_transits": candidate.n_transits,
                    "n_expected_transits": candidate.n_expected_transits,
                    "snr": candidate.snr,
                    "sde": candidate.sde,
                    "power": candidate.power,
                    "fap": candidate.fap,
                    "sectors": ";".join(str(s) for s in candidate.sectors),
                    "quality_flags": ";".join(candidate.quality_flags),
                    "rejection_reason": candidate.rejection_reason or "",
                }
            )
    logger.info("Exported %d candidate(s) to %s", len(candidates), path)
    return path


def write_detection_summary(
    results: list[TCEResult], experiment_name: str, path: Path | str
) -> Path:
    """Writes the run-level detection summary as JSON.

    Args:
        results: Per-target TCE results.
        experiment_name: Name of the TCE run.
        path: Destination JSON file.

    Returns:
        The written file path.
    """
    targets: list[dict[str, Any]] = []
    for result in results:
        accepted = result.accepted
        best = accepted[0] if accepted else None
        targets.append(
            {
                "target_id": result.target_id,
                "n_candidates": len(result.candidates),
                "n_accepted": len(accepted),
                "n_rejected": len(result.rejected),
                "grid": dict(result.grid.provenance),
                "best_candidate": best.to_dict() if best else None,
            }
        )
    summary = {
        "experiment_name": experiment_name,
        "n_targets": len(results),
        "n_accepted_total": sum(
            1
            for result in results
            for c in result.candidates
            if c.status == STATUS_CANDIDATE
        ),
        "targets": targets,
    }
    return write_json(summary, Path(path))
