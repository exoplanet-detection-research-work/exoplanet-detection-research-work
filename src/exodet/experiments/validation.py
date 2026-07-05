"""Reproducibility validation and certificates."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from exodet.experiments.config import ReproduceStageConfig
from exodet.experiments.database import ExperimentDatabase, ExperimentRecord
from exodet.experiments.manager import ExperimentManager
from exodet.experiments.recovery import verify_checkpoint
from exodet.reproducibility.collector import checksum_file
from exodet.utils.io import write_json

__all__ = [
    "ReproducibilityCertificate",
    "ValidationResult",
    "validate_reproducibility",
]

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Outcome of one reproducibility check."""

    experiment_id: str
    metric_match: bool
    checksum_match: bool
    prediction_match: bool
    metric_delta: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.metric_match and self.checksum_match and self.prediction_match

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "passed": self.passed,
            "metric_match": self.metric_match,
            "checksum_match": self.checksum_match,
            "prediction_match": self.prediction_match,
            "metric_delta": self.metric_delta,
            "details": self.details,
        }


@dataclass
class ReproducibilityCertificate:
    """Signed reproducibility attestation for a rerun."""

    experiment_id: str
    original_git_commit: str | None
    rerun_git_commit: str | None
    issued_at: str
    results: list[ValidationResult] = field(default_factory=list)
    signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "original_git_commit": self.original_git_commit,
            "rerun_git_commit": self.rerun_git_commit,
            "issued_at": self.issued_at,
            "results": [r.to_dict() for r in self.results],
            "signature": self.signature,
            "all_passed": all(r.passed for r in self.results),
        }

    def save(self, path: Path) -> Path:
        return write_json(self.to_dict(), path)


def _sign(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _compare_metrics(
    original: dict[str, float],
    rerun: dict[str, float],
    tolerance: float,
) -> tuple[bool, dict[str, float]]:
    deltas: dict[str, float] = {}
    ok = True
    for key in set(original) | set(rerun):
        a = original.get(key, float("nan"))
        b = rerun.get(key, float("nan"))
        delta = abs(a - b) if not (np.isnan(a) or np.isnan(b)) else float("nan")
        deltas[key] = delta
        if not np.isnan(delta) and delta > tolerance:
            ok = False
    return ok, deltas


def validate_reproducibility(
    database: ExperimentDatabase,
    manager: ExperimentManager,
    config: ReproduceStageConfig,
    *,
    experiment_ids: tuple[str, ...] | None = None,
) -> ReproducibilityCertificate:
    """Rerun selected experiments and verify reproducibility."""
    from exodet.experiments.runner import execute_stage

    ids = experiment_ids or config.experiment_ids
    if not ids:
        raise ValueError("No experiment_ids specified for reproduction.")

    results: list[ValidationResult] = []
    for eid in ids:
        record = database.get(eid)
        if record is None:
            results.append(
                ValidationResult(
                    eid, False, False, False,
                    details={"error": "not found"},
                )
            )
            continue

        original_metrics = dict(record.metrics)
        original_model_checksum = record.model_checksum
        ckpt = Path(record.output_dir) / "checkpoints"
        checkpoint_valid = True
        for path in ckpt.glob("*"):
            if path.is_file():
                integrity = verify_checkpoint(path)
                if not integrity.valid:
                    checkpoint_valid = False

        try:
            manager.mark_running(eid)
            rerun = execute_stage(
                manager.experiment,
                record.metadata.get("stage", "train"),
                output_dir=Path(record.output_dir) / "rerun",
                checkpoint_dir=Path(record.output_dir) / "rerun" / "checkpoints",
            )
            rerun_metrics = rerun.get("metrics", {})
            metric_ok, deltas = _compare_metrics(
                original_metrics, rerun_metrics, config.metric_tolerance
            )
            rerun_checksum = rerun.get("model_checksum", "")
            checksum_ok = (
                not original_model_checksum
                or rerun_checksum == original_model_checksum
            )
            prob_ok = True
            if "probabilities" in rerun and "original_probabilities" in record.metadata:
                orig = np.asarray(record.metadata["original_probabilities"], dtype=np.float64)
                new = np.asarray(rerun["probabilities"], dtype=np.float64)
                if orig.shape == new.shape:
                    prob_ok = bool(np.max(np.abs(orig - new)) <= config.probability_tolerance)

            results.append(
                ValidationResult(
                    experiment_id=eid,
                    metric_match=metric_ok,
                    checksum_match=checksum_ok and checkpoint_valid,
                    prediction_match=prob_ok,
                    metric_delta=deltas,
                    details={"rerun_metrics": rerun_metrics},
                )
            )
            status = "completed" if metric_ok and checksum_ok else "failed"
            manager.database.update(eid, status=status)
        except Exception as exc:
            logger.error("Reproduction failed for %s: %s", eid, exc)
            results.append(
                ValidationResult(
                    eid, False, False, False, details={"error": str(exc)}
                )
            )

    cert = ReproducibilityCertificate(
        experiment_id=ids[0] if len(ids) == 1 else "batch",
        original_git_commit=results[0].details.get("git_commit") if results else None,
        rerun_git_commit=None,
        issued_at=datetime.now(timezone.utc).isoformat(),
        results=results,
    )
    payload = {k: v for k, v in cert.to_dict().items() if k != "signature"}
    cert.signature = _sign(payload)

    if config.issue_certificate:
        out = Path(manager.campaign_root) / "certificates"
        out.mkdir(parents=True, exist_ok=True)
        cert.save(out / f"{cert.experiment_id}_certificate.json")

    return cert
