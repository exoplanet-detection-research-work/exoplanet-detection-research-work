"""Experiment index database for campaign tracking."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from exodet.exceptions import PipelineError
from exodet.utils.io import ensure_dir, write_json

__all__ = [
    "ExperimentRecord",
    "ExperimentDatabase",
    "ExperimentStatus",
]

ExperimentStatus = str  # pending | running | completed | failed | interrupted


@dataclass
class ExperimentRecord:
    """One indexed experiment run."""

    experiment_id: str
    name: str
    status: ExperimentStatus = "pending"
    tags: tuple[str, ...] = ()
    parent_id: str | None = None
    template: str | None = None
    config_path: str | None = None
    config_checksum: str = ""
    output_dir: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    git_commit: str | None = None
    dataset_checksum: str = ""
    model_checksum: str = ""
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentRecord":
        return cls(
            experiment_id=str(raw["experiment_id"]),
            name=str(raw.get("name", raw["experiment_id"])),
            status=str(raw.get("status", "pending")),
            tags=tuple(str(t) for t in raw.get("tags", ())),
            parent_id=str(raw["parent_id"]) if raw.get("parent_id") else None,
            template=str(raw["template"]) if raw.get("template") else None,
            config_path=str(raw["config_path"]) if raw.get("config_path") else None,
            config_checksum=str(raw.get("config_checksum", "")),
            output_dir=str(raw.get("output_dir", "")),
            metrics={str(k): float(v) for k, v in dict(raw.get("metrics", {})).items()},
            artifacts={str(k): str(v) for k, v in dict(raw.get("artifacts", {})).items()},
            hardware=dict(raw.get("hardware", {})),
            runtime_seconds=float(raw.get("runtime_seconds", 0.0)),
            git_commit=str(raw["git_commit"]) if raw.get("git_commit") else None,
            dataset_checksum=str(raw.get("dataset_checksum", "")),
            model_checksum=str(raw.get("model_checksum", "")),
            created_at=str(raw.get("created_at", "")),
            updated_at=str(raw.get("updated_at", "")),
            metadata=dict(raw.get("metadata", {})),
            error=str(raw["error"]) if raw.get("error") else None,
        )


class ExperimentDatabase:
    """Thread-safe JSON-backed experiment index."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._records: dict[str, ExperimentRecord] = {}
        self._load()

    def _load(self) -> None:
        if self.path.is_file():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._records = {
                rid: ExperimentRecord.from_dict(rec)
                for rid, rec in dict(data.get("records", {})).items()
            }

    def _save(self) -> None:
        ensure_dir(self.path.parent)
        payload = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(self._records),
            "records": {rid: rec.to_dict() for rid, rec in self._records.items()},
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def register(self, record: ExperimentRecord) -> ExperimentRecord:
        """Insert or replace an experiment record."""
        now = datetime.now(timezone.utc).isoformat()
        if not record.created_at:
            record.created_at = now
        record.updated_at = now
        with self._lock:
            self._records[record.experiment_id] = record
            self._save()
        return record

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        with self._lock:
            return self._records.get(experiment_id)

    def update(self, experiment_id: str, **fields: Any) -> ExperimentRecord:
        with self._lock:
            record = self._records.get(experiment_id)
            if record is None:
                raise PipelineError(f"Unknown experiment id: {experiment_id}")
            data = record.to_dict()
            data.update(fields)
            data["updated_at"] = datetime.now(timezone.utc).isoformat()
            updated = ExperimentRecord.from_dict(data)
            self._records[experiment_id] = updated
            self._save()
            return updated

    def iter_records(self) -> Iterator[ExperimentRecord]:
        with self._lock:
            yield from list(self._records.values())

    def search(
        self,
        *,
        tags: tuple[str, ...] = (),
        status: str | None = None,
        name_contains: str | None = None,
        min_metric: tuple[str, float] | None = None,
        limit: int = 0,
    ) -> list[ExperimentRecord]:
        """Filter indexed experiments."""
        results: list[ExperimentRecord] = []
        for rec in self.iter_records():
            if status and rec.status != status:
                continue
            if tags and not all(t in rec.tags for t in tags):
                continue
            if name_contains and name_contains.lower() not in rec.name.lower():
                continue
            if min_metric is not None:
                key, threshold = min_metric
                if rec.metrics.get(key, float("-inf")) < threshold:
                    continue
            results.append(rec)
        results.sort(key=lambda r: r.updated_at, reverse=True)
        if limit > 0:
            return results[:limit]
        return results

    def bulk_insert(self, records: list[ExperimentRecord]) -> None:
        """Insert many records (for performance benchmarks)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            for rec in records:
                if not rec.created_at:
                    rec.created_at = now
                rec.updated_at = now
                self._records[rec.experiment_id] = rec
            self._save()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._records)
