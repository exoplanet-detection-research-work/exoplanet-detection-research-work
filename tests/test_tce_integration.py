"""Integration, regression, recovery, and edge-case tests of the TCE stage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from exodet.cli.main import main
from exodet.data.base import LightCurve
from exodet.data.serialization import save_light_curve
from exodet.tce import (
    InjectionRecoveryExperiment,
    TCEPipeline,
    TCESearchConfig,
    inject_box_transit,
    load_tce_config,
    make_noise_light_curve,
)
from exodet.exceptions import ConfigurationError
from exodet.visualization.tce import generate_tce_figures


def fast_tce_config(**overrides: object) -> TCESearchConfig:
    """A quick-to-run TCE configuration for tests."""
    raw: dict[str, object] = {
        "experiment_name": "tce_test",
        "seed": 42,
        "grid": {
            "name": "bls_auto",
            "params": {
                "min_period_days": 0.5,
                "max_period_days": 9.0,
                "oversample": 1.0,
                "min_duration_days": 0.05,
                "max_duration_days": 0.25,
                "n_durations": 3,
            },
        },
        "peaks": {
            "name": "prominence",
            "params": {"threshold_sigma": 5.0, "max_candidates": 10},
        },
    }
    raw.update(overrides)
    return TCESearchConfig.from_dict(raw)


@pytest.fixture(scope="module")
def pipeline() -> TCEPipeline:
    return TCEPipeline(fast_tce_config())


def injected_curve(
    period: float = 2.7,
    depth: float = 0.004,
    duration: float = 0.12,
    epoch: float = 0.9,
    noise: float = 5e-4,
    n_points: int = 15_000,
    seed: int = 1,
) -> LightCurve:
    """A noise curve with one injected transit signal."""
    base = make_noise_light_curve(n_points=n_points, noise_level=noise, seed=seed)
    return inject_box_transit(base, period, duration, depth, epoch)


class TestRecovery:
    def test_recovers_injected_signal(self, pipeline: TCEPipeline) -> None:
        result = pipeline.run(injected_curve())
        assert len(result.accepted) == 1
        best = result.accepted[0]
        assert best.period_days == pytest.approx(2.7, rel=0.01)
        assert best.meta["rank"] == 1

    def test_regression_pinned_values(self, pipeline: TCEPipeline) -> None:
        """Regression guard: pinned seed must reproduce known values."""
        result = pipeline.run(injected_curve())
        best = result.accepted[0]
        assert best.period_days == pytest.approx(2.6973, abs=0.005)
        assert 25.0 < best.sde < 50.0
        assert abs(best.depth - 0.004) / 0.004 < 0.35
        assert best.n_transits == 8
        assert best.fap < 1e-20
        # Harmonics of the true signal are detected and labelled.
        harmonic_periods = [
            c.period_days
            for c in result.candidates
            if c.status == "rejected_harmonic"
        ]
        assert any(abs(p - 1.35) < 0.05 for p in harmonic_periods)  # P/2

    def test_short_period_planet(self, pipeline: TCEPipeline) -> None:
        curve = injected_curve(period=0.62, duration=0.05, depth=0.005, seed=5)
        result = pipeline.run(curve)
        assert result.accepted
        assert result.accepted[0].period_days == pytest.approx(0.62, rel=0.01)

    def test_long_period_planet(self, pipeline: TCEPipeline) -> None:
        # ~20.8 d baseline: P = 8 d gives only 2-3 transits.
        curve = injected_curve(period=8.0, duration=0.2, depth=0.006, seed=6)
        result = pipeline.run(curve)
        assert result.accepted
        assert result.accepted[0].period_days == pytest.approx(8.0, rel=0.01)

    def test_shallow_transit(self, pipeline: TCEPipeline) -> None:
        curve = injected_curve(depth=0.0008, noise=3e-4, seed=7)
        result = pipeline.run(curve)
        assert result.accepted
        assert result.accepted[0].depth == pytest.approx(0.0008, rel=0.5)

    def test_multiple_planets_both_recovered(self, pipeline: TCEPipeline) -> None:
        base = make_noise_light_curve(n_points=15_000, noise_level=3e-4, seed=8)
        curve = inject_box_transit(base, 2.1, 0.1, 0.006, 0.4)
        curve = inject_box_transit(curve, 3.7, 0.15, 0.004, 1.2)
        result = pipeline.run(curve)
        periods = sorted(c.period_days for c in result.accepted)
        assert len(periods) >= 2
        assert periods[0] == pytest.approx(2.1, rel=0.01)
        assert periods[1] == pytest.approx(3.7, rel=0.01)


class TestEdgeCases:
    def test_constant_flux_yields_no_candidates(self, pipeline: TCEPipeline) -> None:
        curve = LightCurve(
            target_id="CONST",
            time=np.arange(15_000) * (2.0 / (60 * 24)),
            flux=np.ones(15_000),
        )
        result = pipeline.run(curve)
        assert result.candidates == []

    @pytest.mark.parametrize("seed", [0, 1, 3])
    def test_pure_noise_yields_no_accepted_candidates(
        self, pipeline: TCEPipeline, seed: int
    ) -> None:
        result = pipeline.run(
            make_noise_light_curve(n_points=15_000, noise_level=1e-3, seed=seed)
        )
        assert result.accepted == []

    def test_nan_cadences_tolerated(self, pipeline: TCEPipeline) -> None:
        curve = injected_curve(seed=9)
        flux = curve.flux.copy()
        flux[::50] = np.nan
        curve.flux = flux
        result = pipeline.run(curve)
        assert result.accepted
        assert result.accepted[0].period_days == pytest.approx(2.7, rel=0.01)

    def test_observational_gaps_tolerated(self, pipeline: TCEPipeline) -> None:
        curve = injected_curve(seed=10)
        keep = np.ones(len(curve), dtype=bool)
        keep[5000:7000] = False  # ~2.8-day gap
        gapped = LightCurve(
            target_id=curve.target_id,
            time=np.where(curve.time > curve.time[7000], curve.time + 1.0, curve.time)[
                keep
            ],
            flux=curve.flux[keep],
            flux_err=None if curve.flux_err is None else curve.flux_err[keep],
        )
        result = pipeline.run(gapped)
        assert result.accepted
        assert result.accepted[0].period_days == pytest.approx(2.7, rel=0.01)


class TestInjectionRecoveryExperiment:
    def test_strong_signals_fully_recovered(self, pipeline: TCEPipeline) -> None:
        experiment = InjectionRecoveryExperiment(pipeline)
        base = make_noise_light_curve(n_points=12_000, noise_level=3e-4, seed=11)
        summary = experiment.run(
            [base], periods=[1.9, 3.3], durations=[0.1], depths=[0.005, 0.01], seed=0
        )
        assert summary.n_trials == 4
        assert summary.recovery_rate == 1.0
        assert summary.recall == 1.0
        assert summary.precision == 1.0
        assert summary.median_period_error_rel < 0.01
        assert summary.median_depth_error_rel < 0.5

    def test_undetectable_signals_not_recovered(self, pipeline: TCEPipeline) -> None:
        experiment = InjectionRecoveryExperiment(pipeline)
        base = make_noise_light_curve(n_points=12_000, noise_level=2e-3, seed=12)
        summary = experiment.run(
            [base], periods=[2.3], durations=[0.1], depths=[5e-5], seed=0
        )
        assert summary.recovery_rate == 0.0

    def test_efficiency_curves_and_persistence(
        self, pipeline: TCEPipeline, tmp_path: Path
    ) -> None:
        # Strict exact-period matching: a marginal detection at a
        # harmonic of the hopeless 50 ppm injection must not count.
        experiment = InjectionRecoveryExperiment(pipeline, accept_harmonics=False)
        base = make_noise_light_curve(n_points=12_000, noise_level=5e-4, seed=13)
        summary = experiment.run(
            [base], periods=[2.5], durations=[0.1], depths=[5e-5, 0.008], seed=0
        )
        # Deep signal recovered, hopeless one not: efficiency separates them.
        assert summary.efficiency_by_depth["0.008"] == 1.0
        assert summary.efficiency_by_depth["5e-05"] == 0.0
        assert set(summary.efficiency_by_duration) == {"0.1"}
        assert summary.efficiency_by_snr  # populated

        json_path = summary.save(
            tmp_path / "injection.json", csv_path=tmp_path / "trials.csv"
        )
        assert json_path.is_file()
        csv_text = (tmp_path / "trials.csv").read_text(encoding="utf-8")
        assert csv_text.count("\n") == summary.n_trials + 1  # header + rows


class TestConfig:
    def test_defaults_fill_missing_sections(self) -> None:
        config = TCESearchConfig.from_dict({"experiment_name": "x"})
        assert config.grid.name == "bls_auto"
        assert config.ranking.name == "metric"

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="phased"):
            TCESearchConfig.from_dict({"experiment_name": "x", "phased": {}})

    def test_missing_name_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="experiment_name"):
            TCESearchConfig.from_dict({})

    def test_load_with_overrides(self, tmp_path: Path) -> None:
        path = tmp_path / "tce.yaml"
        path.write_text("experiment_name: t\nseed: 1\n", encoding="utf-8")
        config = load_tce_config(path, overrides=["seed=99", "n_figure_targets=0"])
        assert config.seed == 99
        assert config.n_figure_targets == 0


class TestRunnerAndCli:
    def _write_setup(self, tmp_path: Path) -> Path:
        processed = tmp_path / "processed"
        for index, (period, depth) in enumerate([(2.7, 0.004), (4.1, 0.006)]):
            curve = injected_curve(period=period, depth=depth, seed=20 + index)
            curve.target_id = f"TIC {9000 + index}"
            save_light_curve(curve, processed / f"tic_{9000 + index}.npz")
        config_path = tmp_path / "tce.yaml"
        config_path.write_text(
            f"""\
experiment_name: tce_cli_test
seed: 42
paths:
  processed_dir: {processed}
  figure_dir: {tmp_path / 'figures'}
  log_dir: {tmp_path / 'logs'}
  report_dir: {tmp_path / 'reports'}
logging:
  level: INFO
  to_file: false
grid:
  name: bls_auto
  params:
    min_period_days: 0.5
    max_period_days: 9.0
    oversample: 1.0
    min_duration_days: 0.05
    max_duration_days: 0.25
    n_durations: 3
""",
            encoding="utf-8",
        )
        return config_path

    def test_cli_tce_end_to_end(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_path = self._write_setup(tmp_path)
        assert main(["tce", "-c", str(config_path)]) == 0
        out = capsys.readouterr().out
        assert "2 accepted candidate(s) across 2 target(s)" in out

        reports = tmp_path / "reports"
        assert (reports / "tce_candidates.json").is_file()
        assert (reports / "tce_detection_summary.json").is_file()
        csv_text = (reports / "tce_candidates.csv").read_text(encoding="utf-8")
        assert "TIC 9000" in csv_text and "TIC 9001" in csv_text

        figure_names = {p.name for p in (tmp_path / "figures").iterdir()}
        for kind in (
            "bls_periodogram",
            "bls_power_spectrum",
            "transit_markers",
            "top_candidate",
        ):
            assert f"tic_9000_{kind}.png" in figure_names
            assert f"tic_9000_{kind}.pdf" in figure_names

    def test_cli_tce_no_input_fails_cleanly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_path = tmp_path / "tce.yaml"
        config_path.write_text(
            f"experiment_name: t\npaths:\n  processed_dir: {tmp_path / 'empty'}\n"
            "logging:\n  to_file: false\n",
            encoding="utf-8",
        )
        assert main(["tce", "-c", str(config_path)]) == 1
        assert "No processed light curves" in capsys.readouterr().err


class TestFigures:
    def test_figures_for_result_without_candidates(self, tmp_path: Path) -> None:
        pipeline = TCEPipeline(fast_tce_config())
        curve = make_noise_light_curve(n_points=12_000, noise_level=1e-3, seed=0)
        result = pipeline.run(curve)
        written = generate_tce_figures(curve, result, tmp_path)
        names = {p.name for p in written}
        assert any("bls_periodogram" in n for n in names)
        assert not any("top_candidate" in n for n in names)
