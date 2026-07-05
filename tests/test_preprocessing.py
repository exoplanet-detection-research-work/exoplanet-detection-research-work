"""Unit and edge-case tests for every preprocessing step."""

from __future__ import annotations

import numpy as np
import pytest

from exodet.data.base import LightCurve
from exodet.exceptions import PipelineError
from exodet.preprocessing import (
    PREPROCESSORS,
    GapDetector,
    GapInterpolator,
    NaNRemover,
    Normalizer,
    QualityFlagFilter,
    QualityMetrics,
    SectorStitcher,
    SigmaClipper,
    WotanDetrender,
    estimate_cdpp,
)
from tests.conftest import make_synthetic_tess_curve


def _simple_curve(
    flux: np.ndarray,
    time: np.ndarray | None = None,
    **meta: object,
) -> LightCurve:
    time = np.arange(len(flux), dtype=np.float64) * 0.01 if time is None else time
    return LightCurve(
        target_id="UNIT", time=time, flux=flux.astype(np.float64), meta=dict(meta)
    )


class TestRegistration:
    EXPECTED = (
        "quality_filter",
        "nan_removal",
        "sector_stitch",
        "gap_detect",
        "gap_interpolate",
        "wotan_detrend",
        "sigma_clip",
        "normalize",
        "quality_metrics",
    )

    def test_all_steps_registered(self) -> None:
        for name in self.EXPECTED:
            assert name in PREPROCESSORS


class TestImmutability:
    @pytest.mark.parametrize(
        "step",
        [
            QualityFlagFilter(),
            NaNRemover(strategy="fill_interpolate"),
            SectorStitcher(),
            GapDetector(),
            GapInterpolator(),
            SigmaClipper(),
            QualityMetrics(),
        ],
        ids=lambda s: type(s).__name__,
    )
    def test_input_never_modified(self, tess_curve: LightCurve, step) -> None:
        time_before = tess_curve.time.copy()
        flux_before = tess_curve.flux.copy()
        history_before = list(tess_curve.history)
        meta_keys_before = set(tess_curve.meta)

        result = step.apply(tess_curve)

        assert result is not tess_curve
        np.testing.assert_array_equal(tess_curve.time, time_before)
        np.testing.assert_array_equal(tess_curve.flux, flux_before)
        assert tess_curve.history == history_before
        assert set(tess_curve.meta) == meta_keys_before
        assert len(result.history) == len(history_before) + 1


class TestQualityFlagFilter:
    def test_removes_flagged_cadences(self) -> None:
        flux = np.ones(10)
        quality = np.zeros(10, dtype=np.int64)
        quality[[2, 5]] = 128
        curve = _simple_curve(flux, quality=quality)
        result = QualityFlagFilter(bitmask=128).apply(curve)
        assert len(result) == 8
        assert (result.meta["quality"] & 128).sum() == 0

    def test_preset_default_keeps_straylight(self) -> None:
        quality = np.array([0, 128, 2048, 0], dtype=np.int64)
        curve = _simple_curve(np.ones(4), quality=quality)
        result = QualityFlagFilter(bitmask="default").apply(curve)
        assert len(result) == 3  # only manual_exclude removed

    def test_preset_hardest_removes_all_flagged(self) -> None:
        quality = np.array([0, 1, 2048, 4095], dtype=np.int64)
        curve = _simple_curve(np.ones(4), quality=quality)
        result = QualityFlagFilter(bitmask="hardest").apply(curve)
        assert len(result) == 1

    def test_missing_quality_is_noop_with_provenance(self) -> None:
        curve = _simple_curve(np.ones(5))
        result = QualityFlagFilter().apply(curve)
        assert len(result) == 5
        assert len(result.history) == 1

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(PipelineError, match="preset"):
            QualityFlagFilter(bitmask="banana")

    def test_negative_bitmask_raises(self) -> None:
        with pytest.raises(PipelineError, match=">= 0"):
            QualityFlagFilter(bitmask=-1)

    def test_all_flagged_raises(self) -> None:
        curve = _simple_curve(np.ones(3), quality=np.full(3, 128, dtype=np.int64))
        with pytest.raises(PipelineError, match="all"):
            QualityFlagFilter(bitmask=128).apply(curve)


class TestNaNRemover:
    def test_drop_removes_nonfinite(self) -> None:
        flux = np.array([1.0, np.nan, 1.0, np.inf, 1.0])
        result = NaNRemover(strategy="drop").apply(_simple_curve(flux))
        assert len(result) == 3
        assert np.isfinite(result.flux).all()
        assert result.meta["nan_removal"] == {"n_dropped": 2, "n_filled": 0}

    def test_fill_median(self) -> None:
        flux = np.array([1.0, np.nan, 3.0])
        result = NaNRemover(strategy="fill_median").apply(_simple_curve(flux))
        assert len(result) == 3
        assert result.flux[1] == pytest.approx(2.0)

    def test_fill_interpolate(self) -> None:
        flux = np.array([1.0, np.nan, 3.0])
        result = NaNRemover(strategy="fill_interpolate").apply(_simple_curve(flux))
        assert result.flux[1] == pytest.approx(2.0)

    def test_nonfinite_time_always_dropped(self) -> None:
        time = np.array([0.0, np.nan, 0.2])
        flux = np.array([1.0, 1.0, 1.0])
        result = NaNRemover(strategy="fill_median").apply(
            _simple_curve(flux, time=time)
        )
        assert len(result) == 2

    def test_flux_err_nans_filled_with_median(self) -> None:
        curve = LightCurve(
            target_id="U",
            time=np.arange(4, dtype=np.float64),
            flux=np.ones(4),
            flux_err=np.array([0.1, np.nan, 0.3, 0.1]),
        )
        result = NaNRemover().apply(curve)
        assert np.isfinite(result.flux_err).all()
        assert result.flux_err[1] == pytest.approx(0.1)

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(PipelineError, match="strategy"):
            NaNRemover(strategy="magic")

    def test_all_nan_flux_raises(self) -> None:
        curve = _simple_curve(np.full(5, np.nan))
        with pytest.raises(PipelineError):
            NaNRemover(strategy="fill_median").apply(curve)


class TestSectorStitcher:
    def test_normalizes_sector_offsets(self) -> None:
        n = 200
        flux = np.concatenate([np.full(n, 1000.0), np.full(n, 2000.0)])
        sector = np.repeat([1, 2], n)
        curve = _simple_curve(flux, sector=sector)
        result = SectorStitcher().apply(curve)
        assert result.flux == pytest.approx(np.ones(2 * n))
        assert result.meta["sector_medians"] == {1: 1000.0, 2: 2000.0}

    def test_sorts_by_time(self) -> None:
        time = np.array([3.0, 1.0, 2.0])
        curve = _simple_curve(np.array([3.0, 1.0, 2.0]), time=time)
        result = SectorStitcher().apply(curve)
        np.testing.assert_array_equal(result.time, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result.flux, np.array([1.0, 2.0, 3.0]) / 2.0)

    def test_no_sector_meta_treated_as_single_sector(self) -> None:
        result = SectorStitcher().apply(_simple_curve(np.full(50, 500.0)))
        assert result.flux == pytest.approx(np.ones(50))

    def test_zero_median_raises(self) -> None:
        with pytest.raises(PipelineError, match="median"):
            SectorStitcher().apply(_simple_curve(np.zeros(10)))

    def test_stitch_classmethod_combines_curves(self) -> None:
        first = make_synthetic_tess_curve(n_sectors=1, defects=False, seed=1)
        second = make_synthetic_tess_curve(n_sectors=1, defects=False, seed=2)
        second = LightCurve(
            target_id=first.target_id,
            time=second.time + 30.0,
            flux=second.flux * 3.0,
            flux_err=second.flux_err,
            meta={"sector": 7},
        )
        first.meta["sector"] = 3
        stitched = SectorStitcher.stitch([first, second])
        assert len(stitched) == len(first) + len(second)
        assert np.all(np.diff(stitched.time) >= 0)
        assert set(stitched.meta["sector_medians"]) == {3, 7}
        provenance = stitched.meta["stitched_from"]
        assert [entry["sector"] for entry in provenance] == [3, 7]

    def test_stitch_empty_raises(self) -> None:
        with pytest.raises(PipelineError, match="empty"):
            SectorStitcher.stitch([])

    def test_stitch_mixed_targets_raises(self) -> None:
        a = _simple_curve(np.ones(5))
        b = LightCurve(target_id="OTHER", time=a.time, flux=a.flux)
        with pytest.raises(PipelineError, match="different targets"):
            SectorStitcher.stitch([a, b])


class TestGapDetector:
    def test_detects_gap(self) -> None:
        time = np.concatenate([np.arange(50) * 0.01, 2.0 + np.arange(50) * 0.01])
        curve = _simple_curve(np.ones(100), time=time)
        result = GapDetector(factor=5.0).apply(curve)
        gaps = result.meta["gaps"]
        assert len(gaps) == 1
        assert gaps[0]["start_time"] == pytest.approx(0.49)
        assert gaps[0]["end_time"] == pytest.approx(2.0)
        assert gaps[0]["start_index"] == 49

    def test_no_gaps(self) -> None:
        curve = _simple_curve(np.ones(100))
        result = GapDetector().apply(curve)
        assert result.meta["gaps"] == []

    def test_invalid_factor_raises(self) -> None:
        with pytest.raises(PipelineError, match="factor"):
            GapDetector(factor=0.0)


class TestGapInterpolator:
    @staticmethod
    def _gappy_curve() -> LightCurve:
        time = np.concatenate([np.arange(100) * 0.01, 1.3 + np.arange(100) * 0.01])
        flux = 1.0 + 0.1 * np.sin(time)
        return _simple_curve(flux, time=time)

    @pytest.mark.parametrize("method", ["linear", "spline"])
    def test_fills_short_gap(self, method: str) -> None:
        curve = self._gappy_curve()
        result = GapInterpolator(method=method, max_gap_days=0.5).apply(curve)
        assert len(result) > len(curve)
        assert np.all(np.diff(result.time) > 0)
        interpolated = result.meta["interpolated"]
        assert interpolated.sum() == len(result) - len(curve)
        # Interpolated flux must track the smooth underlying signal.
        synthetic = result.flux[interpolated]
        expected = 1.0 + 0.1 * np.sin(result.time[interpolated])
        np.testing.assert_allclose(synthetic, expected, atol=5e-3)

    def test_long_gap_not_bridged(self) -> None:
        curve = self._gappy_curve()  # gap of ~0.31 d
        result = GapInterpolator(method="linear", max_gap_days=0.1).apply(curve)
        assert len(result) == len(curve)
        assert not result.meta["interpolated"].any()

    def test_none_method_is_provenance_only(self) -> None:
        curve = self._gappy_curve()
        result = GapInterpolator(method="none").apply(curve)
        assert len(result) == len(curve)
        assert result.history == ["GapInterpolator(method=none)"]

    def test_reuses_detector_metadata(self) -> None:
        curve = GapDetector().apply(self._gappy_curve())
        result = GapInterpolator(method="linear").apply(curve)
        assert result.meta["interpolated"].any()

    def test_per_cadence_meta_extended(self) -> None:
        curve = self._gappy_curve()
        curve.meta["quality"] = np.zeros(len(curve), dtype=np.int64)
        curve.meta["sector"] = np.repeat([1, 2], 100).astype(np.int64)
        result = GapInterpolator(method="linear").apply(curve)
        assert result.meta["quality"].shape == (len(result),)
        assert result.meta["sector"].shape == (len(result),)
        assert (result.meta["quality"][result.meta["interpolated"]] == 0).all()

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(PipelineError, match="method"):
            GapInterpolator(method="quadratic")


class TestSigmaClipper:
    def test_removes_outliers(self) -> None:
        rng = np.random.default_rng(1)
        flux = 1.0 + rng.normal(0.0, 1e-3, 1000)
        flux[[100, 500]] += 0.5
        result = SigmaClipper(sigma=5.0).apply(_simple_curve(flux))
        assert len(result) == 998
        assert result.meta["clipped_time"].size == 2

    def test_clip_lower_false_protects_transits(self) -> None:
        rng = np.random.default_rng(2)
        flux = 1.0 + rng.normal(0.0, 1e-4, 500)
        flux[100] += 0.1  # cosmic ray, should go
        flux[200] -= 0.1  # transit-like dip, should stay
        result = SigmaClipper(sigma=5.0, clip_lower=False).apply(_simple_curve(flux))
        assert len(result) == 499
        assert result.meta["clipped_flux"][0] == pytest.approx(flux[100])

    def test_iterative_clipping_converges(self) -> None:
        rng = np.random.default_rng(3)
        flux = 1.0 + rng.normal(0.0, 1e-4, 2000)
        flux[:20] += np.linspace(0.01, 1.0, 20)  # staircase of outliers
        few = SigmaClipper(sigma=5.0, max_iterations=1).apply(_simple_curve(flux))
        many = SigmaClipper(sigma=5.0, max_iterations=10).apply(_simple_curve(flux))
        assert len(many) <= len(few)
        assert len(many) == 1980

    def test_constant_flux_untouched(self) -> None:
        result = SigmaClipper().apply(_simple_curve(np.ones(50)))
        assert len(result) == 50

    def test_invalid_params_raise(self) -> None:
        with pytest.raises(PipelineError, match="sigma"):
            SigmaClipper(sigma=0.0)
        with pytest.raises(PipelineError, match="max_iterations"):
            SigmaClipper(max_iterations=0)


class TestNormalizer:
    def test_minmax_range(self, clean_tess_curve: LightCurve) -> None:
        result = Normalizer(method="minmax").apply(clean_tess_curve)
        assert result.flux.min() == pytest.approx(0.0)
        assert result.flux.max() == pytest.approx(1.0)

    def test_median_centering(self, clean_tess_curve: LightCurve) -> None:
        result = Normalizer(method="median").apply(clean_tess_curve)
        assert np.median(result.flux) == pytest.approx(1.0)

    def test_zscore_moments(self, clean_tess_curve: LightCurve) -> None:
        result = Normalizer(method="zscore").apply(clean_tess_curve)
        assert np.mean(result.flux) == pytest.approx(0.0, abs=1e-12)
        assert np.std(result.flux) == pytest.approx(1.0)

    def test_stats_recorded_and_invertible(self, clean_tess_curve: LightCurve) -> None:
        result = Normalizer(method="zscore").apply(clean_tess_curve)
        stats = result.meta["normalization"]["stats"]
        recovered = result.flux * stats["std"] + stats["mean"]
        np.testing.assert_allclose(recovered, clean_tess_curve.flux)

    def test_flux_err_scaled(self, clean_tess_curve: LightCurve) -> None:
        result = Normalizer(method="median").apply(clean_tess_curve)
        stats = result.meta["normalization"]["stats"]
        np.testing.assert_allclose(
            result.flux_err, clean_tess_curve.flux_err / abs(stats["median"])
        )

    @pytest.mark.parametrize("method", ["minmax", "zscore"])
    def test_constant_flux_raises(self, method: str) -> None:
        with pytest.raises(PipelineError, match="constant"):
            Normalizer(method=method).apply(_simple_curve(np.ones(10)))

    def test_zero_median_raises(self) -> None:
        flux = np.concatenate([-np.ones(5), np.zeros(1), np.ones(5)])
        with pytest.raises(PipelineError, match="median"):
            Normalizer(method="median").apply(_simple_curve(flux))

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(PipelineError, match="method"):
            Normalizer(method="rank")


class TestWotanDetrender:
    @pytest.mark.parametrize("method", ["biweight", "median", "lowess"])
    def test_removes_variability(self, method: str, clean_tess_curve: LightCurve) -> None:
        result = WotanDetrender(
            method=method, window_length_days=0.5
        ).apply(clean_tess_curve)
        raw_scatter = np.std(clean_tess_curve.flux / np.median(clean_tess_curve.flux))
        detrended_scatter = np.std(result.flux)
        assert detrended_scatter < 0.5 * raw_scatter
        assert np.median(result.flux) == pytest.approx(1.0, abs=1e-3)
        assert result.meta["trend"].shape == result.flux.shape

    def test_transit_survives_detrending(self, clean_tess_curve: LightCurve) -> None:
        result = WotanDetrender(window_length_days=0.5).apply(clean_tess_curve)
        assert result.flux.min() < 1.0 - 0.004  # depth 0.008, allow half

    def test_flux_err_rescaled(self, clean_tess_curve: LightCurve) -> None:
        result = WotanDetrender(window_length_days=0.5).apply(clean_tess_curve)
        assert np.median(result.flux_err) == pytest.approx(3e-4, rel=0.1)

    def test_invalid_params_raise(self) -> None:
        with pytest.raises(PipelineError, match="method"):
            WotanDetrender(method="magic")
        with pytest.raises(PipelineError, match="window_length_days"):
            WotanDetrender(window_length_days=-1.0)


class TestQualityMetrics:
    KEYS = {
        "rms_ppm",
        "cdpp_ppm",
        "variance",
        "skewness",
        "kurtosis",
        "duty_cycle",
        "missing_fraction",
        "n_points",
        "timespan_days",
        "median_cadence_days",
    }

    def test_metric_keys_and_sanity(self, clean_tess_curve: LightCurve) -> None:
        result = QualityMetrics().apply(clean_tess_curve)
        metrics = result.meta["quality_metrics"]
        assert set(metrics) == self.KEYS
        assert metrics["rms_ppm"] > 0
        assert 0 < metrics["duty_cycle"] <= 1
        assert metrics["missing_fraction"] == pytest.approx(
            1.0 - metrics["duty_cycle"]
        )
        assert metrics["n_points"] == len(clean_tess_curve)

    def test_cdpp_matches_white_noise_expectation(self) -> None:
        rng = np.random.default_rng(4)
        n = 5000
        time = np.arange(n) * (2.0 / (60 * 24))
        flux = 1.0 + rng.normal(0.0, 1e-4, n)
        cdpp = estimate_cdpp(time, flux, duration_hours=1.0)
        # 100 ppm white noise averaged over 30 cadences -> ~18 ppm.
        assert 8.0 < cdpp < 40.0

    def test_cdpp_nan_for_short_curve(self) -> None:
        time = np.arange(5, dtype=np.float64) * 0.001
        cdpp = estimate_cdpp(time, np.ones(5))
        assert np.isnan(cdpp)

    def test_interpolated_cadences_excluded_from_duty_cycle(self) -> None:
        flux = np.ones(100) + np.arange(100) * 1e-5
        curve = _simple_curve(flux)
        without = QualityMetrics().apply(curve)
        curve.meta["interpolated"] = np.zeros(100, dtype=bool)
        curve.meta["interpolated"][:50] = True
        with_mask = QualityMetrics().apply(curve)
        assert (
            with_mask.meta["quality_metrics"]["duty_cycle"]
            < without.meta["quality_metrics"]["duty_cycle"]
        )

    def test_invalid_duration_raises(self) -> None:
        with pytest.raises(PipelineError, match="cdpp_duration_hours"):
            QualityMetrics(cdpp_duration_hours=0.0)
