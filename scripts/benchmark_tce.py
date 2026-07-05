"""Benchmarks the TCE pipeline over 1 / 10 / 100 / 1000 light curves.

Measures wall time, CPU time (and utilization), Python-level peak
memory (tracemalloc), process peak RSS, and per-curve throughput, then
checks scalability against the single-curve baseline. Results are
printed as a table and written to ``outputs/reports/tce_benchmark.json``.

Usage:
    python scripts/benchmark_tce.py [--sizes 1 10 100 1000]
        [--n-points 10000] [--output outputs/reports/tce_benchmark.json]
"""

from __future__ import annotations

import argparse
import json
import resource
import time
import tracemalloc
from pathlib import Path
from typing import Any

from exodet.tce import (
    TCEPipeline,
    TCESearchConfig,
    inject_box_transit,
    make_noise_light_curve,
)
from exodet.utils.seeding import seed_everything


def build_pipeline() -> TCEPipeline:
    """Builds the benchmark pipeline (default components)."""
    config = TCESearchConfig.from_dict(
        {
            "experiment_name": "tce_benchmark",
            "grid": {
                "name": "bls_auto",
                "params": {
                    "min_period_days": 0.5,
                    "max_period_days": 6.0,
                    "oversample": 1.0,
                    "min_duration_days": 0.05,
                    "max_duration_days": 0.25,
                    "n_durations": 3,
                },
            },
        }
    )
    return TCEPipeline(config)


def make_curves(n: int, n_points: int) -> list:
    """Builds ``n`` synthetic curves, each with one injected transit."""
    curves = []
    for index in range(n):
        base = make_noise_light_curve(
            target_id=f"BENCH-{index:04d}",
            n_points=n_points,
            noise_level=5e-4,
            seed=index,
        )
        period = 1.5 + (index % 7) * 0.45
        curves.append(inject_box_transit(base, period, 0.1, 0.004, 0.3))
    return curves


def benchmark_batch(pipeline: TCEPipeline, curves: list) -> dict[str, Any]:
    """Runs the pipeline over a batch and collects all measurements."""
    tracemalloc.start()
    cpu_start = time.process_time()
    wall_start = time.perf_counter()

    n_accepted = 0
    for curve in curves:
        n_accepted += len(pipeline.run(curve).accepted)

    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS

    return {
        "n_curves": len(curves),
        "n_accepted": n_accepted,
        "wall_seconds": round(wall, 3),
        "cpu_seconds": round(cpu, 3),
        "cpu_utilization": round(cpu / wall, 3) if wall > 0 else None,
        "seconds_per_curve": round(wall / len(curves), 4),
        "tracemalloc_peak_mb": round(peak_bytes / 1e6, 2),
        "max_rss_mb": round(max_rss / 1e6, 2),
    }


def main() -> None:
    """Runs the benchmark and writes the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 10, 100, 1000])
    parser.add_argument("--n-points", type=int, default=10_000)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/reports/tce_benchmark.json")
    )
    args = parser.parse_args()

    seed_everything(42)
    pipeline = build_pipeline()
    # Warm-up: first call pays import/JIT/cache costs.
    pipeline.run(make_curves(1, args.n_points)[0])

    rows: list[dict[str, Any]] = []
    header = (
        f"{'n':>6} {'wall [s]':>10} {'s/curve':>9} {'CPU util':>9} "
        f"{'py-peak [MB]':>13} {'RSS [MB]':>9} {'accepted':>9}"
    )
    print(header)
    print("-" * len(header))
    for size in args.sizes:
        curves = make_curves(size, args.n_points)
        row = benchmark_batch(pipeline, curves)
        rows.append(row)
        print(
            f"{row['n_curves']:>6} {row['wall_seconds']:>10.2f} "
            f"{row['seconds_per_curve']:>9.3f} {row['cpu_utilization']:>9.2f} "
            f"{row['tracemalloc_peak_mb']:>13.1f} {row['max_rss_mb']:>9.0f} "
            f"{row['n_accepted']:>9}"
        )

    baseline = rows[0]["seconds_per_curve"]
    report = {
        "n_points_per_curve": args.n_points,
        "results": rows,
        "scalability": {
            str(row["n_curves"]): round(row["seconds_per_curve"] / baseline, 3)
            for row in rows
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
