"""Performance regression tests for the TCE stage.

Bounds are generous (5-10x typical laptop timings) to stay green on
slow CI hardware while catching complexity regressions. The full
benchmark (1/10/100/1000 curves, memory and CPU profiling) lives in
``scripts/benchmark_tce.py``.
"""

from __future__ import annotations

import time

import numpy as np

from exodet.tce import TCEPipeline, inject_box_transit, make_noise_light_curve
from tests.test_tce_integration import fast_tce_config


def _make_curves(n: int, n_points: int = 12_000) -> list:
    curves = []
    for index in range(n):
        base = make_noise_light_curve(
            target_id=f"PERF-{index}", n_points=n_points, noise_level=5e-4, seed=index
        )
        curves.append(inject_box_transit(base, 2.0 + 0.3 * index % 5, 0.1, 0.005, 0.7))
    return curves


class TestTCEThroughput:
    def test_single_curve_runtime(self) -> None:
        pipeline = TCEPipeline(fast_tce_config())
        curve = _make_curves(1)[0]
        pipeline.run(curve)  # warm-up (imports, caches)
        start = time.perf_counter()
        result = pipeline.run(curve)
        elapsed = time.perf_counter() - start
        print(f"single curve ({len(curve):,} cadences): {elapsed:.2f} s")
        assert result.accepted
        assert elapsed < 10.0

    def test_ten_curves_scale_linearly(self) -> None:
        pipeline = TCEPipeline(fast_tce_config())
        curves = _make_curves(10)
        pipeline.run(curves[0])  # warm-up

        start = time.perf_counter()
        single = time.perf_counter()
        pipeline.run(curves[0])
        single = time.perf_counter() - single

        results = [pipeline.run(curve) for curve in curves]
        total = time.perf_counter() - start
        print(f"10 curves: {total:.2f} s total, {single:.2f} s single")
        assert all(r.periodogram is not None for r in results)
        assert total < 60.0
        # Linear-ish scaling: 10 curves must cost well under 30x one curve.
        assert total < max(30.0 * single, 10.0)

    def test_periodogram_memory_footprint(self) -> None:
        # The periodogram must be O(n_frequencies), not O(n_freq x n_dur).
        pipeline = TCEPipeline(fast_tce_config())
        curve = _make_curves(1)[0]
        result = pipeline.run(curve)
        n = len(result.periodogram)
        arrays = (
            result.periodogram.power,
            result.periodogram.depth,
            result.periodogram.duration,
            result.periodogram.transit_time,
        )
        total_bytes = sum(a.nbytes for a in arrays) + result.periodogram.periods.nbytes
        assert total_bytes < 8 * n * 10  # a handful of float64 arrays only
        assert np.isfinite(result.periodogram.power).all()
