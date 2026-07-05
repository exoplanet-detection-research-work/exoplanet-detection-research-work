"""Artifact organization and cleanup policies."""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from exodet.experiments.config import ArtifactsStageConfig
from exodet.experiments.database import ExperimentRecord
from exodet.utils.io import ensure_dir, write_json

__all__ = ["ArtifactIndex", "organize_artifacts", "apply_cleanup_policy"]

logger = logging.getLogger(__name__)

ARTIFACT_CATEGORIES = (
    "models",
    "figures",
    "reports",
    "catalogs",
    "logs",
    "datasets",
    "benchmarks",
)


@dataclass
class ArtifactIndex:
    """Catalog of artifact paths for one experiment."""

    experiment_id: str
    paths: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"experiment_id": self.experiment_id, "paths": self.paths}

    def save(self, output_dir: Path) -> Path:
        path = output_dir / "artifacts.json"
        write_json(self.to_dict(), path)
        return path


def _classify(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth", ".joblib", ".json"} and "model" in name:
        return "models"
    if suffix in {".png", ".pdf", ".svg"}:
        return "figures"
    if suffix in {".html", ".md", ".csv"} and "report" in name:
        return "reports"
    if "catalog" in name:
        return "catalogs"
    if suffix == ".log" or "log" in name:
        return "logs"
    if suffix == ".npz":
        return "datasets"
    if "benchmark" in name:
        return "benchmarks"
    if suffix in {".html", ".md", ".csv", ".json"}:
        return "reports"
    if suffix in {".png", ".pdf"}:
        return "figures"
    return "reports"


def organize_artifacts(record: ExperimentRecord, *, config: ArtifactsStageConfig) -> ArtifactIndex:
    """Move artifacts into category subdirectories under the experiment root."""
    root = Path(record.output_dir)
    index = ArtifactIndex(experiment_id=record.experiment_id)
    if not config.enabled or not config.organize:
        return index

    for category in ARTIFACT_CATEGORIES:
        ensure_dir(root / category)
        index.paths[category] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ARTIFACT_CATEGORIES for part in path.parts):
            continue
        if path.name in {"experiment.json", "state.json", "artifacts.json"}:
            continue
        category = _classify(path)
        dest = root / category / path.name
        if dest != path:
            if dest.is_file():
                dest.unlink()
            shutil.move(str(path), str(dest))
        index.paths.setdefault(category, []).append(str(dest))

    index.save(root)
    return index


def apply_cleanup_policy(
    records: list[ExperimentRecord],
    policy: dict[str, Any],
) -> list[str]:
    """Apply retention cleanup and return removed paths."""
    if not policy.get("enabled", False):
        return []
    max_age_days = float(policy.get("max_age_days", 30))
    keep_status = set(policy.get("keep_status", ["completed"]))
    max_bytes = int(policy.get("max_total_bytes", 0))
    cutoff = time.time() - max_age_days * 86400
    removed: list[str] = []

    for rec in records:
        if rec.status in keep_status:
            continue
        root = Path(rec.output_dir)
        if not root.is_dir():
            continue
        mtime = root.stat().st_mtime
        if mtime < cutoff:
            shutil.rmtree(root, ignore_errors=True)
            removed.append(str(root))

    if max_bytes > 0:
        dirs = sorted(
            (Path(r.output_dir) for r in records if Path(r.output_dir).is_dir()),
            key=lambda p: p.stat().st_mtime,
        )
        total = sum(
            f.stat().st_size for d in dirs for f in d.rglob("*") if f.is_file()
        )
        for d in dirs:
            if total <= max_bytes:
                break
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d, ignore_errors=True)
            total -= size
            removed.append(str(d))

    return removed
