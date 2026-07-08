"""Benchmarks the representation pipeline over 1..10000 candidates."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from exodet.representation import RepresentationConfig, RepresentationPipeline
from exodet.representation.cache import RepresentationCache
from exodet.representation.containers import RepresentationDataset
from exodet.tce import inject_box_transit, make_noise_light_curve
from exodet.tce.candidate import TransitCandidate
from exodet.utils.process_metrics import process_rss_bytes
from exodet.utils.seeding import seed_everything


def _make_candidate(
    target_id: str, period: float, epoch: float, duration: float, depth: float
) -> TransitCandidate:
    slug = target_id.replace(" ", "_")
    return TransitCandidate(
        candidate_id=f"{slug}-01",
        target_id=target_id,
        sectors=(),
        period_days=period,
        epoch_days=epoch,
        duration_days=duration,
        depth=depth,
        depth_err=depth * 0.1,
        n_transits=8,
        n_expected_transits=10,
        snr=50.0,
        sde=30.0,
        power=50.0,
        fap=1e-20,
    )


def _benchmark_config(n_bins_global: int = 201, n_bins_local: int = 81) -> RepresentationConfig:
    return RepresentationConfig.from_dict(
        {
            "experiment_name": "rep_benchmark",
            "n_figure_samples": 0,
            "global_view": {
                "name": "global",
                "params": {"n_bins": n_bins_global, "max_empty_fraction": 0.6},
            },
            "local_view": {
                "name": "local",
                "params": {"n_bins": n_bins_local, "max_empty_fraction": 0.6},
            },
            "cache": {"enabled": False},
            "splitting": {
                "name": "star",
                "params": {"validation_fraction": 0.0, "test_fraction": 0.0},
            },
        }
    )


def _make_pairs(n: int, n_points: int = 10_000):
    pairs = []
    for index in range(n):
        period = 1.2 + 0.05 * (index % 10)
        target_id = f"BENCH-{index:05d}"
        injected = inject_box_transit(
            make_noise_light_curve(
                target_id=target_id,
                n_points=n_points,
                noise_level=5e-4,
                seed=index,
            ),
            period_days=period,
            duration_days=0.1,
            depth=0.005,
            epoch_days=0.3,
        )
        pairs.append(
            (
                injected,
                _make_candidate(target_id, period, 0.3, 0.1, 0.005),
            )
        )
    return pairs


def benchmark_batch(
    pipeline: RepresentationPipeline, pairs: list, cache: RepresentationCache | None
) -> dict:
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    samples = []
    for curve, candidate in pairs:
        samples.append(pipeline.build_sample(curve, candidate))
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    rss = process_rss_bytes() or 0

    tmp = Path("outputs/reports/_bench_tmp.npz")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    RepresentationDataset(samples).save(tmp)
    disk_bytes = tmp.stat().st_size + tmp.with_suffix(".json").stat().st_size
    tmp.unlink(missing_ok=True)
    tmp.with_suffix(".json").unlink(missing_ok=True)

    row = {
        "n_candidates": len(pairs),
        "wall_seconds": round(wall, 3),
        "cpu_seconds": round(cpu, 3),
        "cpu_utilization": round(cpu / wall, 3) if wall > 0 else None,
        "seconds_per_sample": round(wall / max(len(pairs), 1), 4),
        "max_rss_mb": round(rss / (1024**2), 1),
        "disk_kb_per_sample": round(disk_bytes / max(len(pairs), 1) / 1024, 1),
    }
    if cache is not None:
        row["cache_hit_rate"] = round(cache.stats["hit_rate"], 3)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 10, 100, 1000, 10000])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/representation_benchmark.json"),
    )
    args = parser.parse_args()

    seed_everything(42)
    config = _benchmark_config()
    pipeline = RepresentationPipeline(config)
    # Warm-up on one pair.
    warm = _make_pairs(1)
    if warm:
        pipeline.build_sample(*warm[0])

    rows = []
    header = (
        f"{'n':>6} {'wall [s]':>10} {'s/sample':>9} {'CPU':>6} "
        f"{'RSS [MB]':>9} {'disk [KB/s]':>12}"
    )
    print(header)
    print("-" * len(header))

    for size in args.sizes:
        n_points = 8000 if size <= 100 else 4000
        pairs = _make_pairs(size, n_points=n_points)
        if len(pairs) < size:
            print(f"warning: expected {size} pairs, got {len(pairs)}")
        row = benchmark_batch(pipeline, pairs, cache=None)
        rows.append(row)
        print(
            f"{row['n_candidates']:>6} {row['wall_seconds']:>10.2f} "
            f"{row['seconds_per_sample']:>9.3f} {row['cpu_utilization']:>6.2f} "
            f"{row['max_rss_mb']:>9.0f} {row['disk_kb_per_sample']:>12.1f}"
        )

    # Cache benchmark on 100 samples.
    cache_dir = Path("outputs/reports/_rep_cache_bench")
    cache = RepresentationCache(cache_dir, compress=True)
    cache.clear()
    cached_pipeline = RepresentationPipeline(
        _benchmark_config(), cache=cache
    )
    pairs100 = _make_pairs(100, n_points=8000)
    for curve, candidate in pairs100:
        cached_pipeline.build_sample(curve, candidate)
    start = time.perf_counter()
    for curve, candidate in pairs100:
        cached_pipeline.build_sample(curve, candidate)
    cache_wall = time.perf_counter() - start

    baseline = rows[0]["seconds_per_sample"] if rows else 1.0
    report = {
        "results": rows,
        "scalability": {
            str(r["n_candidates"]): round(r["seconds_per_sample"] / baseline, 3)
            for r in rows
        },
        "cache_100_reads_seconds": round(cache_wall, 3),
        "cache_hit_rate": cache.stats["hit_rate"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {args.output}")
    cache.clear()


if __name__ == "__main__":
    main()
