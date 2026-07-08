"""Database performance benchmarks."""

from __future__ import annotations

import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from exodet.experiments.database import ExperimentDatabase, ExperimentRecord
from exodet.utils.process_metrics import process_rss_bytes

__all__ = ["PerformanceBenchmark", "benchmark_database_scales"]


@dataclass
class PerformanceBenchmark:
    """Performance metrics at one scale."""

    n_records: int
    insert_seconds: float
    query_seconds: float
    artifact_lookup_seconds: float
    database_bytes: int
    peak_memory_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_records": self.n_records,
            "insert_seconds": self.insert_seconds,
            "query_seconds": self.query_seconds,
            "artifact_lookup_seconds": self.artifact_lookup_seconds,
            "database_bytes": self.database_bytes,
            "peak_memory_bytes": self.peak_memory_bytes,
        }


def _peak_memory_bytes() -> int:
    return process_rss_bytes() or 0


def _make_records(n: int) -> list[ExperimentRecord]:
    tmp_root = Path(tempfile.gettempdir()) / "exodet_bench"
    return [
        ExperimentRecord(
            experiment_id=f"exp_{i:06d}",
            name=f"benchmark_{i}",
            status="completed",
            tags=("benchmark",),
            metrics={"roc_auc": 0.5 + (i % 50) / 100.0, "accuracy": 0.8},
            artifacts={"checkpoint": str(tmp_root / f"ckpt_{i}.pt")},
            runtime_seconds=float(i % 100),
        )
        for i in range(n)
    ]


def benchmark_database_scales(
    output_dir: Path,
    scales: tuple[int, ...] = (100, 500, 1000),
) -> list[PerformanceBenchmark]:
    """Benchmark insert/query/lookup at multiple record counts."""
    results: list[PerformanceBenchmark] = []
    for n in scales:
        db_path = output_dir / f"bench_{n}.json"
        if db_path.is_file():
            db_path.unlink()
        db = ExperimentDatabase(db_path)
        records = _make_records(n)

        t0 = time.perf_counter()
        db.bulk_insert(records)
        insert_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(10):
            db.search(tags=("benchmark",), min_metric=("roc_auc", 0.6), limit=50)
        query_seconds = (time.perf_counter() - t0) / 10.0

        t0 = time.perf_counter()
        for rec in records[: min(100, n)]:
            db.get(rec.experiment_id)
            _ = rec.artifacts.get("checkpoint")
        artifact_lookup_seconds = (time.perf_counter() - t0) / min(100, n)

        db_bytes = db_path.stat().st_size if db_path.is_file() else 0
        results.append(
            PerformanceBenchmark(
                n_records=n,
                insert_seconds=insert_seconds,
                query_seconds=query_seconds,
                artifact_lookup_seconds=artifact_lookup_seconds,
                database_bytes=db_bytes,
                peak_memory_bytes=_peak_memory_bytes(),
            )
        )
    return results
