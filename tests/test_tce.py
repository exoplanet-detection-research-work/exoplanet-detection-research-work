"""Unit tests for every TCE module."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from exodet.data.base import LightCurve
from exodet.exceptions import DataError, PipelineError
from exodet.tce import (
    GRID_GENERATORS,
    HARMONIC_REJECTERS,
    METRICS_COMPUTERS,
    PEAK_DETECTORS,
    RANKERS,
    SEARCH_ENGINES,
    VALIDATORS,
    AstropyBLSEngine,
    BLSGridGenerator,
    CompositeRanker,
    MetricRanker,
    Periodogram,
    PeriodRatioHarmonicRejecter,
    PhysicalValidator,
    ProminencePeakDetector,
    TransitCandidate,
    inject_box_transit,
    make_noise_light_curve,
)
from exodet.tce.candidate import (
    STATUS_CANDIDATE,
    STATUS_REJECTED_HARMONIC,
    STATUS_REJECTED_VALIDATION,
    SearchGrid,
    load_candidates,
    save_candidates,
)
from exodet.tce.metrics import gaussian_fap, sde_of_peak


def make_candidate(**overrides: object) -> TransitCandidate:
    """Builds a physically valid candidate; fields overridable."""
    defaults: dict[str, object] = {
        "candidate_id": "TIC_1-01",
        "target_id": "TIC 1",
        "sectors": (1, 2),
        "period_days": 2.5,
        "epoch_days": 0.7,
        "duration_days": 0.1,
        "depth": 0.005,
        "depth_err": 0.0005,
        "n_transits": 5,
        "n_expected_transits": 6,
        "snr": 10.0,
        "sde": 12.0,
        "power": 50.0,
        "fap": 1e-10,
    }
    defaults.update(overrides)
    return TransitCandidate(**defaults)  # type: ignore[arg-type]


def make_grid(min_period: float = 0.5, max_period: float = 10.0) -> SearchGrid:
    """Builds a simple search grid for validation tests."""
    return SearchGrid(
        periods=np.linspace(max_period, min_period, 100),
        durations=np.array([0.1]),
        provenance={"baseline_days": 20.0},
    )


class TestRegistration:
    def test_all_tce_registries_populated(self) -> None:
        assert "bls_auto" in GRID_GENERATORS
        assert "astropy_bls" in SEARCH_ENGINES
        assert "prominence" in PEAK_DETECTORS
        assert "standard" in METRICS_COMPUTERS
        assert "physical" in VALIDATORS
        assert "period_ratio" in HARMONIC_REJECTERS
        assert "metric" in RANKERS
        assert "composite" in RANKERS


class TestGridGenerator:
    def test_generates_valid_grid_with_provenance(self) -> None:
        curve = make_noise_light_curve(n_points=10_000, seed=0)
        grid = BLSGridGenerator(
            min_period_days=0.5, max_period_days=5.0, oversample=2.0
        ).generate(curve)
        # Grid must never search below the requested minimum period.
        assert grid.min_period >= 0.5
        assert grid.min_period == pytest.approx(0.5, rel=1e-3)
        assert grid.max_period == pytest.approx(5.0, rel=1e-3)
        assert np.all(np.diff(grid.frequencies) > 0)
        # Uniform frequency spacing.
        assert np.allclose(np.diff(grid.frequencies), np.diff(grid.frequencies)[0])
        for key in (
            "baseline_days",
            "median_cadence_days",
            "nyquist_period_days",
            "frequency_spacing_per_day",
            "n_frequencies",
            "durations_days",
        ):
            assert key in grid.provenance

    def test_nyquist_violation_raises(self) -> None:
        curve = make_noise_light_curve(n_points=1000, cadence_days=0.5, seed=0)
        generator = BLSGridGenerator(min_period_days=0.6, min_duration_days=0.1,
                                     max_duration_days=0.2)
        with pytest.raises(PipelineError, match="cadence limit"):
            generator.generate(curve)

    def test_max_period_clamped_to_baseline(self) -> None:
        curve = make_noise_light_curve(n_points=5000, seed=0)  # ~6.9 d baseline
        grid = BLSGridGenerator(
            min_period_days=0.5, max_period_days=100.0, min_n_transits=2
        ).generate(curve)
        baseline = grid.provenance["baseline_days"]
        assert grid.max_period == pytest.approx(baseline / 2, rel=1e-6)
        assert any("clamped" in note for note in grid.provenance["notes"])

    def test_auto_max_period(self) -> None:
        curve = make_noise_light_curve(n_points=5000, seed=0)
        grid = BLSGridGenerator(min_period_days=0.5, min_n_transits=3).generate(curve)
        assert grid.max_period == pytest.approx(
            grid.provenance["baseline_days"] / 3, rel=1e-6
        )

    def test_explicit_n_frequencies(self) -> None:
        curve = make_noise_light_curve(n_points=5000, seed=0)
        grid = BLSGridGenerator(
            min_period_days=0.5, max_period_days=3.0, n_frequencies=500
        ).generate(curve)
        assert len(grid) == 500

    def test_duration_longer_than_period_raises(self) -> None:
        with pytest.raises(PipelineError, match="cannot outlast"):
            BLSGridGenerator(min_period_days=0.2, max_duration_days=0.3)

    def test_baseline_too_short_raises(self) -> None:
        curve = make_noise_light_curve(n_points=100, seed=0)  # ~0.14 d
        with pytest.raises(PipelineError, match="baseline"):
            BLSGridGenerator(min_period_days=0.5).generate(curve)

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(PipelineError, match="oversample"):
            BLSGridGenerator(oversample=0.5)
        with pytest.raises(PipelineError, match="min_period_days"):
            BLSGridGenerator(min_period_days=-1.0)
        with pytest.raises(PipelineError, match="n_durations"):
            BLSGridGenerator(n_durations=0)


class TestSearchEngine:
    def test_periodogram_arrays_aligned(self) -> None:
        curve = make_noise_light_curve(n_points=5000, seed=1)
        grid = BLSGridGenerator(
            min_period_days=0.5, max_period_days=3.0, oversample=1.0
        ).generate(curve)
        pgram = AstropyBLSEngine(objective="snr").search(curve, grid)
        n = len(grid)
        for array in (
            pgram.power,
            pgram.depth,
            pgram.depth_snr,
            pgram.duration,
            pgram.transit_time,
            pgram.log_likelihood,
        ):
            assert array.shape == (n,)
        assert np.isfinite(pgram.power).all()
        assert pgram.meta["n_points"] == 5000

    def test_nan_cadences_ignored(self) -> None:
        curve = make_noise_light_curve(n_points=5000, seed=2)
        flux = curve.flux.copy()
        flux[::100] = np.nan
        curve = replace_flux(curve, flux)
        grid = BLSGridGenerator(
            min_period_days=0.5, max_period_days=3.0, oversample=1.0
        ).generate(curve)
        pgram = AstropyBLSEngine().search(curve, grid)
        assert pgram.meta["n_points"] == 5000 - 50

    def test_too_few_points_raises(self) -> None:
        curve = LightCurve(
            target_id="TINY", time=np.arange(5, dtype=float), flux=np.ones(5)
        )
        grid = make_grid()
        with pytest.raises(PipelineError, match="at least 10"):
            AstropyBLSEngine().search(curve, grid)

    def test_invalid_objective_raises(self) -> None:
        with pytest.raises(PipelineError, match="objective"):
            AstropyBLSEngine(objective="magic")


def replace_flux(curve: LightCurve, flux: np.ndarray) -> LightCurve:
    """Helper: swap the flux array of a curve without provenance noise."""
    return LightCurve(
        target_id=curve.target_id,
        time=curve.time,
        flux=flux,
        flux_err=curve.flux_err,
        mission=curve.mission,
        meta=dict(curve.meta),
    )


def make_periodogram(
    n: int = 2000, peak_index: int = 700, peak_height: float = 20.0, seed: int = 0
) -> Periodogram:
    """Builds a synthetic periodogram with one injected peak."""
    rng = np.random.default_rng(seed)
    frequencies = np.linspace(0.1, 2.0, n)
    power = rng.normal(1.0, 0.2, n)
    power[peak_index] += peak_height
    power[peak_index - 1] += 0.5 * peak_height
    power[peak_index + 1] += 0.5 * peak_height
    zeros = np.zeros(n)
    return Periodogram(
        periods=1.0 / frequencies,
        power=power,
        depth=zeros,
        depth_snr=zeros,
        duration=zeros,
        transit_time=zeros,
        log_likelihood=zeros,
        objective="snr",
        meta={"grid": {"baseline_days": 20.0}, "target_id": "SYNTH"},
    )


class TestPeakDetector:
    def test_finds_injected_peak(self) -> None:
        pgram = make_periodogram()
        indices = ProminencePeakDetector(threshold_sigma=5.0).detect(pgram)
        assert 700 in indices.tolist()

    def test_max_candidates_cap(self) -> None:
        pgram = make_periodogram()
        power = pgram.power.copy()
        for idx in (200, 400, 1000, 1300, 1600):
            power[idx] += 15.0
        pgram = replace(pgram, power=power)
        indices = ProminencePeakDetector(threshold_sigma=5.0, max_candidates=3).detect(
            pgram
        )
        assert len(indices) == 3
        # Strongest peak first.
        assert indices[0] == 700

    def test_flat_spectrum_yields_no_peaks(self) -> None:
        pgram = replace(make_periodogram(), power=np.ones(2000))
        assert ProminencePeakDetector().detect(pgram).size == 0

    def test_absolute_floor_applies(self) -> None:
        pgram = make_periodogram(peak_height=20.0)
        indices = ProminencePeakDetector(
            threshold_sigma=5.0, min_power=1000.0
        ).detect(pgram)
        assert indices.size == 0

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(PipelineError, match="threshold_sigma"):
            ProminencePeakDetector(threshold_sigma=0.0)
        with pytest.raises(PipelineError, match="max_candidates"):
            ProminencePeakDetector(max_candidates=0)


class TestDetectionMetrics:
    def test_sde_standardizes_peak(self) -> None:
        power = np.zeros(1000)
        power[500] = 10.0
        sde = sde_of_peak(power, 500, robust=False)
        assert sde == pytest.approx(
            (10.0 - power.mean()) / power.std(), rel=1e-12
        )

    def test_sde_nan_for_flat_spectrum(self) -> None:
        assert math.isnan(sde_of_peak(np.ones(100), 50))

    def test_fap_monotonic_in_sde(self) -> None:
        faps = [gaussian_fap(sde, 1e4) for sde in (3.0, 5.0, 7.0, 10.0)]
        assert all(a > b for a, b in zip(faps, faps[1:]))
        assert 0.0 <= faps[-1] < faps[0] <= 1.0

    def test_fap_extremes(self) -> None:
        assert gaussian_fap(50.0, 1e4) == pytest.approx(0.0, abs=1e-30)
        assert gaussian_fap(0.0, 1e6) == pytest.approx(1.0)
        assert math.isnan(gaussian_fap(math.nan, 1e4))


class TestTransitCandidate:
    def test_immutability(self) -> None:
        candidate = make_candidate()
        with pytest.raises(AttributeError):
            candidate.period_days = 3.0  # type: ignore[misc]

    def test_with_status_preserves_original(self) -> None:
        candidate = make_candidate()
        rejected = candidate.with_status(
            STATUS_REJECTED_VALIDATION, "too shallow", stage="test"
        )
        assert candidate.status == STATUS_CANDIDATE
        assert rejected.status == STATUS_REJECTED_VALIDATION
        assert rejected.rejection_reason == "too shallow"
        assert rejected.history == ("test",)

    def test_serialization_round_trip(self, tmp_path: Path) -> None:
        candidates = [
            make_candidate(),
            make_candidate(candidate_id="TIC_1-02", status=STATUS_REJECTED_HARMONIC,
                           rejection_reason="2P of TIC_1-01"),
        ]
        path = save_candidates(candidates, tmp_path / "cands.json")
        loaded = load_candidates(path)
        assert loaded == candidates

    def test_load_malformed_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('[{"nope": 1}]', encoding="utf-8")
        with pytest.raises(DataError, match="Malformed"):
            load_candidates(path)
        with pytest.raises(DataError, match="not found"):
            load_candidates(tmp_path / "missing.json")


class TestPhysicalValidator:
    def test_valid_candidate_passes(self) -> None:
        result = PhysicalValidator().validate([make_candidate()], make_grid())
        assert result[0].status == STATUS_CANDIDATE
        assert result[0].rejection_reason is None

    @pytest.mark.parametrize(
        ("overrides", "expected_reason"),
        [
            ({"n_transits": 1}, "observed transit"),
            ({"n_transits": 2, "n_expected_transits": 10}, "coverage"),
            ({"duration_days": 0.8}, "duration/period"),
            ({"depth": 0.9}, "depth"),
            ({"depth": -0.001}, "depth"),
            ({"period_days": 50.0}, "outside searched range"),
            ({"sde": 3.0}, "SDE"),
            ({"snr": 1.0}, "SNR"),
            ({"period_days": math.nan}, "non-finite"),
        ],
    )
    def test_rejection_criteria(
        self, overrides: dict[str, object], expected_reason: str
    ) -> None:
        candidate = make_candidate(**overrides)
        result = PhysicalValidator().validate([candidate], make_grid())
        assert result[0].status == STATUS_REJECTED_VALIDATION
        assert expected_reason in result[0].rejection_reason

    def test_all_failures_reported(self) -> None:
        candidate = make_candidate(sde=1.0, snr=1.0, n_transits=1)
        result = PhysicalValidator().validate([candidate], make_grid())
        reason = result[0].rejection_reason
        assert "SDE" in reason and "SNR" in reason and "transit" in reason

    def test_rejected_candidates_retained(self) -> None:
        candidates = [make_candidate(), make_candidate(sde=1.0)]
        result = PhysicalValidator().validate(candidates, make_grid())
        assert len(result) == 2

    def test_invalid_thresholds_raise(self) -> None:
        with pytest.raises(PipelineError, match="min_coverage"):
            PhysicalValidator(min_coverage=2.0)


class TestHarmonicRejecter:
    def _family(self) -> list[TransitCandidate]:
        return [
            make_candidate(candidate_id="A", period_days=2.5, power=100.0),
            make_candidate(candidate_id="B", period_days=1.25, power=60.0),  # P/2
            make_candidate(candidate_id="C", period_days=5.0, power=50.0),  # 2P
            make_candidate(candidate_id="D", period_days=7.5, power=40.0),  # 3P
            make_candidate(candidate_id="E", period_days=3.76, power=30.0),  # alias 3/2
            make_candidate(candidate_id="F", period_days=1.7, power=20.0),  # unrelated
        ]

    def test_rejects_integer_harmonics_and_aliases(self) -> None:
        result = PeriodRatioHarmonicRejecter(tolerance=0.01).reject(self._family())
        by_id = {c.candidate_id: c for c in result}
        assert by_id["A"].status == STATUS_CANDIDATE
        assert by_id["F"].status == STATUS_CANDIDATE
        for cid, ratio in (("B", "1/2"), ("C", "2/1"), ("D", "3/1"), ("E", "3/2")):
            assert by_id[cid].status == STATUS_REJECTED_HARMONIC
            assert ratio in by_id[cid].rejection_reason
            assert "A" in by_id[cid].rejection_reason

    def test_strongest_survives_regardless_of_order(self) -> None:
        family = list(reversed(self._family()))
        result = PeriodRatioHarmonicRejecter().reject(family)
        accepted = {c.candidate_id for c in result if c.status == STATUS_CANDIDATE}
        assert accepted == {"A", "F"}

    def test_previously_rejected_pass_through(self) -> None:
        rejected = make_candidate(
            candidate_id="X", status=STATUS_REJECTED_VALIDATION,
            rejection_reason="low SDE",
        )
        result = PeriodRatioHarmonicRejecter().reject([make_candidate(), rejected])
        assert result[1].status == STATUS_REJECTED_VALIDATION

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(PipelineError, match="tolerance"):
            PeriodRatioHarmonicRejecter(tolerance=0.9)
        with pytest.raises(PipelineError, match="metric"):
            PeriodRatioHarmonicRejecter(metric="depth")


class TestRanking:
    def _candidates(self) -> list[TransitCandidate]:
        return [
            make_candidate(candidate_id="A", snr=5.0, sde=20.0, power=10.0),
            make_candidate(candidate_id="B", snr=15.0, sde=10.0, power=20.0),
            make_candidate(
                candidate_id="C", snr=1.0, sde=1.0, power=1.0,
                status=STATUS_REJECTED_VALIDATION, rejection_reason="x",
            ),
        ]

    def test_metric_ranker_orders_by_metric(self) -> None:
        ranked = MetricRanker(metric="snr").rank(self._candidates())
        assert [c.candidate_id for c in ranked[:2]] == ["B", "A"]
        assert ranked[0].meta["rank"] == 1
        assert ranked[1].meta["rank"] == 2
        assert "rank" not in ranked[2].meta  # rejected: unranked

    def test_metric_ranker_sde(self) -> None:
        ranked = MetricRanker(metric="sde").rank(self._candidates())
        assert ranked[0].candidate_id == "A"

    def test_composite_weights_change_order(self) -> None:
        sde_heavy = CompositeRanker(weights={"sde": 1.0}).rank(self._candidates())
        snr_heavy = CompositeRanker(weights={"snr": 1.0}).rank(self._candidates())
        assert sde_heavy[0].candidate_id == "A"
        assert snr_heavy[0].candidate_id == "B"

    def test_composite_score_recorded(self) -> None:
        ranked = CompositeRanker().rank(self._candidates())
        assert 0.0 <= ranked[0].meta["ranking_score"] <= 1.0

    def test_no_accepted_candidates(self) -> None:
        only_rejected = [self._candidates()[2]]
        assert CompositeRanker().rank(only_rejected) == only_rejected

    def test_invalid_configuration_raises(self) -> None:
        with pytest.raises(PipelineError, match="metric"):
            MetricRanker(metric="depth")
        with pytest.raises(PipelineError, match="Unknown metrics"):
            CompositeRanker(weights={"magic": 1.0})
        with pytest.raises(PipelineError, match="positive"):
            CompositeRanker(weights={"snr": 0.0})


class TestInjection:
    def test_injects_expected_depth_and_count(self) -> None:
        base = make_noise_light_curve(n_points=10_000, noise_level=1e-5, seed=3)
        injected = inject_box_transit(
            base, period_days=2.0, duration_days=0.1, depth=0.01, epoch_days=0.5
        )
        params = injected.meta["injection"]
        # ~duration/period of all points fall in transit.
        expected = 0.1 / 2.0 * len(base)
        assert params["n_in_transit"] == pytest.approx(expected, rel=0.05)
        in_transit_flux = injected.flux[injected.flux < 0.995]
        assert in_transit_flux.size == params["n_in_transit"]
        assert np.median(in_transit_flux) == pytest.approx(0.99, abs=1e-3)
        # Base curve untouched.
        assert base.flux.min() > 0.99
        assert injected.history[-1].startswith("inject_box_transit")

    def test_invalid_parameters_raise(self) -> None:
        base = make_noise_light_curve(n_points=100, seed=0)
        with pytest.raises(PipelineError, match="depth"):
            inject_box_transit(base, 2.0, 0.1, 1.5, 0.0)
        with pytest.raises(PipelineError, match="duration"):
            inject_box_transit(base, 2.0, 3.0, 0.01, 0.0)
        with pytest.raises(PipelineError, match="period"):
            inject_box_transit(base, -1.0, 0.1, 0.01, 0.0)

    def test_noise_curve_reproducible(self) -> None:
        first = make_noise_light_curve(seed=7, n_points=1000)
        second = make_noise_light_curve(seed=7, n_points=1000)
        np.testing.assert_array_equal(first.flux, second.flux)
