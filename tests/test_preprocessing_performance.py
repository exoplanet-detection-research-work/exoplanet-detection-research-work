"""Performance regression tests for the preprocessing steps.

Bounds are deliberately generous (roughly 5-10x typical laptop
timings) so the suite stays green on slow CI machines while still
catching accidental O(n^2) regressions or Python-loop rewrites.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from exodet.data.base import LightCurve
from exodet.preprocessing import (
    GapDetector,
    GapInterpolator,
    NaNRemover,
    Normalizer,
    QualityFlagFilter,
    QualityMetrics,
    SectorStitcher,
    SigmaClipper,
    WotanDetrender,
)

N_LARGE = 200_000


@pytest.fixture(scope="module")
def large_curve() -> LightCurve:
    """A 200k-cadence curve with defects, ~2 sectors of 20s data."""
    rng = np.random.default_rng(0)
    time_arr = np.arange(N_LARGE) * (20.0 / 86400.0)
    time_arr[N_LARGE // 2 :] += 1.0  # downlink gap
    flux = 1000.0 * (1.0 + rng.normal(0.0, 3e-4, N_LARGE))
    flux[rng.choice(N_LARGE, 200, replace=False)] = np.nan
    flux[rng.choice(N_LARGE, 100, replace=False)] += 50.0
    quality = np.zeros(N_LARGE, dtype=np.int64)
    quality[rng.choice(N_LARGE, 500, replace=False)] = 128
    sector = np.repeat([1, 2], N_LARGE // 2)
    return LightCurve(
        target_id="PERF",
        time=time_arr,
        flux=flux,
        flux_err=np.full(N_LARGE, 0.3),
        meta={"quality": quality, "sector": sector},
    )


def _timed(step, curve: LightCurve) -> tuple[LightCurve, float]:
    start = time.perf_counter()
    result = step.apply(curve)
    elapsed = time.perf_counter() - start
    print(f"{type(step).__name__}: {elapsed * 1e3:.1f} ms on {len(curve):,} cadences")
    return result, elapsed


class TestVectorizedStepThroughput:
    def test_quality_filter(self, large_curve: LightCurve) -> None:
        _, elapsed = _timed(QualityFlagFilter(bitmask="hard"), large_curve)
        assert elapsed < 1.0

    def test_nan_removal(self, large_curve: LightCurve) -> None:
        _, elapsed = _timed(NaNRemover(strategy="drop"), large_curve)
        assert elapsed < 1.0

    def test_sector_stitch(self, large_curve: LightCurve) -> None:
        clean = NaNRemover().apply(large_curve)
        _, elapsed = _timed(SectorStitcher(), clean)
        assert elapsed < 2.0

    def test_gap_detect(self, large_curve: LightCurve) -> None:
        _, elapsed = _timed(GapDetector(), large_curve)
        assert elapsed < 1.0

    def test_gap_interpolate(self, large_curve: LightCurve) -> None:
        clean = NaNRemover().apply(large_curve)
        _, elapsed = _timed(GapInterpolator(method="linear"), clean)
        assert elapsed < 2.0

    def test_sigma_clip(self, large_curve: LightCurve) -> None:
        clean = NaNRemover().apply(large_curve)
        _, elapsed = _timed(SigmaClipper(sigma=5.0), clean)
        assert elapsed < 3.0

    def test_normalize(self, large_curve: LightCurve) -> None:
        clean = NaNRemover().apply(large_curve)
        _, elapsed = _timed(Normalizer(method="zscore"), clean)
        assert elapsed < 1.0

    def test_quality_metrics(self, large_curve: LightCurve) -> None:
        clean = NaNRemover().apply(large_curve)
        _, elapsed = _timed(QualityMetrics(), clean)
        assert elapsed < 5.0


class TestDetrendingThroughput:
    def test_wotan_median_20k(self) -> None:
        rng = np.random.default_rng(1)
        n = 20_000
        curve = LightCurve(
            target_id="PERF-DETREND",
            time=np.arange(n) * (2.0 / (60 * 24)),
            flux=1.0 + 0.01 * np.sin(np.arange(n) * 0.01) + rng.normal(0, 1e-4, n),
        )
        _, elapsed = _timed(
            WotanDetrender(method="median", window_length_days=0.5), curve
        )
        assert elapsed < 30.0

    def test_memory_frugality_no_input_growth(self, large_curve: LightCurve) -> None:
        # Steps must not attach large buffers to the *input* object.
        n_meta_before = len(large_curve.meta)
        NaNRemover().apply(large_curve)
        QualityFlagFilter().apply(large_curve)
        assert len(large_curve.meta) == n_meta_before
