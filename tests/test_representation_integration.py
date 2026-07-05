"""Integration, regression, and edge-case tests of the representation stage."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from exodet.cli.main import main
from exodet.data.base import LightCurve
from exodet.data.serialization import save_light_curve
from exodet.exceptions import ConfigurationError, DataError, PipelineError
from exodet.representation import (
    RepresentationConfig,
    RepresentationPipeline,
    load_representation_config,
    run_dataset_build,
)
from exodet.tce import TCEPipeline, inject_box_transit, make_noise_light_curve
from exodet.tce.candidate import save_candidates
from tests.conftest import make_synthetic_tess_curve
from tests.representation_helpers import (
    make_representation_pair,
    wrong_period_candidate,
)
from tests.test_tce import make_candidate
from tests.test_tce_integration import fast_tce_config


def fast_representation_config(**overrides: object) -> RepresentationConfig:
    """A quick representation config for tests (small bin counts)."""
    raw: dict[str, object] = {
        "experiment_name": "rep_test",
        "seed": 42,
        "n_figure_samples": 0,
        "global_view": {
            "name": "global",
            "params": {"n_bins": 201, "max_empty_fraction": 0.6},
        },
        "local_view": {
            "name": "local",
            "params": {"n_bins": 81, "max_empty_fraction": 0.6},
        },
        "splitting": {
            "name": "star",
            "params": {
                "validation_fraction": 0.0,
                "test_fraction": 0.0,
                "seed": 0,
            },
        },
        "cache": {"enabled": False},
    }
    raw.update(overrides)
    return RepresentationConfig.from_dict(raw)


def _build_tce_catalog(tmp_path: Path, n_targets: int = 3) -> tuple[Path, Path]:
    """Writes processed curves and a TCE catalog for integration tests."""
    processed = tmp_path / "processed"
    report = tmp_path / "reports"
    processed.mkdir()
    report.mkdir()
    tce_pipe = TCEPipeline(fast_tce_config())
    candidates = []
    for index in range(n_targets):
        tic = 9000 + index
        injected = inject_box_transit(
            make_noise_light_curve(
                target_id=f"TIC {tic}",
                n_points=15_000,
                noise_level=5e-4,
                seed=index,
            ),
            period_days=2.0 + 0.3 * index,
            duration_days=0.1,
            depth=0.005,
            epoch_days=0.5,
        )
        save_light_curve(injected, processed / f"tic_{tic}.npz")
        result = tce_pipe.run(injected)
        candidates.extend(result.accepted)
    save_candidates(candidates, report / "tce_candidates.json")
    return processed, report


class TestPipeline:
    def test_builds_sample_from_candidate(self) -> None:
        curve, candidate = make_representation_pair(seed=0)
        pipeline = RepresentationPipeline(fast_representation_config())
        sample = pipeline.build_sample(curve, candidate)
        assert sample.global_view.shape == (201,)
        assert sample.local_view.shape == (81,)
        assert len(sample.features) == len(sample.feature_names)
        assert any("phase_fold" in entry for entry in sample.history)

    def test_regression_pinned_shapes(self) -> None:
        curve, candidate = make_representation_pair(seed=1)
        sample = RepresentationPipeline(fast_representation_config()).build_sample(
            curve, candidate
        )
        assert sample.global_view.min() <= -0.5  # astronet normalization
        assert sample.features.shape[0] >= 25
        assert sample.meta["n_folded_cadences"] > 1000


class TestEdgeCases:
    def test_single_transit_curve(self) -> None:
        curve, candidate = make_representation_pair(
            seed=2, period_days=8.0, n_points=3000
        )
        sample = RepresentationPipeline(fast_representation_config()).build_sample(
            curve, candidate
        )
        assert sample.global_view.size == 201

    def test_gapped_observations(self) -> None:
        curve = make_synthetic_tess_curve(defects=True, n_per_sector=1200, seed=2)
        injected = inject_box_transit(curve, 1.3, 0.1, 0.006, 0.5)
        candidate = make_candidate(
            target_id=curve.target_id,
            period_days=1.3,
            epoch_days=0.5,
            duration_days=0.1,
            depth=0.006,
        )
        sample = RepresentationPipeline(fast_representation_config()).build_sample(
            injected, candidate
        )
        assert np.isfinite(sample.global_view).all()

    def test_shallow_transit(self) -> None:
        curve, candidate = make_representation_pair(
            seed=3, depth=0.0008, noise_level=3e-4
        )
        sample = RepresentationPipeline(fast_representation_config()).build_sample(
            curve, candidate
        )
        assert sample.candidate.depth == pytest.approx(0.0008, rel=1e-6)

    def test_deep_eclipse(self) -> None:
        curve, candidate = make_representation_pair(seed=4, depth=0.15)
        sample = RepresentationPipeline(fast_representation_config()).build_sample(
            curve, candidate
        )
        assert sample.candidate.depth == pytest.approx(0.15, rel=1e-6)

    def test_incorrect_period_still_builds(self) -> None:
        curve, candidate = make_representation_pair(seed=5)
        wrong = wrong_period_candidate(candidate)
        sample = RepresentationPipeline(fast_representation_config()).build_sample(
            curve, wrong
        )
        assert sample.global_view.size == 201

    def test_incorrect_epoch_alignment_corrects(self) -> None:
        curve, candidate = make_representation_pair(seed=6, epoch_days=0.9)
        offset = replace(candidate, epoch_days=0.9 + 0.02 * candidate.duration_days)
        sample = RepresentationPipeline(fast_representation_config()).build_sample(
            curve, offset
        )
        assert abs(sample.meta["epoch_correction_days"]) <= 0.5 * candidate.duration_days

    def test_constant_flux_builds_views(self) -> None:
        curve = LightCurve(
            target_id="CONST",
            time=np.arange(5000) * (2.0 / (60 * 24)),
            flux=np.ones(5000),
        )
        candidate = make_candidate(target_id="CONST", period_days=1.3, depth=0.0)
        sample = RepresentationPipeline(fast_representation_config()).build_sample(
            curve, candidate
        )
        assert sample.global_view.size == 201

    def test_pure_noise_with_manual_candidate(self) -> None:
        base = make_noise_light_curve(n_points=10_000, noise_level=1e-3, seed=7)
        candidate = make_candidate(target_id=base.target_id, depth=0.001)
        sample = RepresentationPipeline(fast_representation_config()).build_sample(
            base, candidate
        )
        assert sample.features.size >= 20

    def test_multiple_sectors(self) -> None:
        curve = make_synthetic_tess_curve(n_sectors=3, n_per_sector=800, seed=8)
        injected = inject_box_transit(curve, 1.3, 0.1, 0.006, 0.5)
        candidate = make_candidate(
            target_id=curve.target_id,
            period_days=1.3,
            epoch_days=0.5,
            duration_days=0.1,
            depth=0.006,
        )
        sample = RepresentationPipeline(fast_representation_config()).build_sample(
            injected, candidate
        )
        assert sample.global_view.size == 201


class TestRunnerAndCli:
    def test_run_dataset_build_end_to_end(self, tmp_path: Path) -> None:
        processed, report = _build_tce_catalog(tmp_path, n_targets=3)
        config_path = tmp_path / "rep.yaml"
        config_path.write_text(
            f"""\
experiment_name: rep_cli
seed: 42
paths:
  processed_dir: {processed}
  interim_dir: {tmp_path / 'interim'}
  figure_dir: {tmp_path / 'figures'}
  log_dir: {tmp_path / 'logs'}
  report_dir: {report}
logging:
  to_file: false
n_figure_samples: 0
global_view:
  name: global
  params:
    n_bins: 201
    max_empty_fraction: 0.6
local_view:
  name: local
  params:
    n_bins: 81
    max_empty_fraction: 0.6
splitting:
  name: star
  params:
    validation_fraction: 0.0
    test_fraction: 0.0
cache:
  enabled: true
  directory: {tmp_path / 'cache'}
""",
            encoding="utf-8",
        )
        splits = run_dataset_build(load_representation_config(config_path))
        assert len(splits.train) >= 1
        dataset_dir = processed / "dataset"
        assert (dataset_dir / "train.npz").is_file()
        assert (dataset_dir / "feature_scaler.json").is_file()
        summary = json.loads((report / "dataset_build_summary.json").read_text())
        assert summary["n_samples"] >= 1

    def test_cli_dataset(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        processed, report = _build_tce_catalog(tmp_path, n_targets=2)
        config_path = tmp_path / "rep.yaml"
        config_path.write_text(
            f"""\
experiment_name: rep_cli
paths:
  processed_dir: {processed}
  interim_dir: {tmp_path / 'interim'}
  figure_dir: {tmp_path / 'figures'}
  log_dir: {tmp_path / 'logs'}
  report_dir: {report}
logging:
  to_file: false
n_figure_samples: 0
global_view:
  name: global
  params:
    n_bins: 101
    max_empty_fraction: 0.7
local_view:
  name: local
  params:
    n_bins: 41
    max_empty_fraction: 0.7
splitting:
  name: star
  params:
    validation_fraction: 0.0
    test_fraction: 0.0
cache:
  enabled: false
""",
            encoding="utf-8",
        )
        assert main(["dataset", "-c", str(config_path)]) == 0
        assert "Dataset build:" in capsys.readouterr().out

    def test_missing_catalog_raises(self, tmp_path: Path) -> None:
        config = fast_representation_config(
            paths={
                "processed_dir": str(tmp_path / "processed"),
                "report_dir": str(tmp_path / "reports"),
            }
        )
        with pytest.raises(PipelineError, match="TCE catalog"):
            run_dataset_build(config)


class TestConfig:
    def test_defaults(self) -> None:
        config = RepresentationConfig.from_dict({"experiment_name": "x"})
        assert config.folding.name == "standard"
        assert config.global_view.name == "global"

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="phased"):
            RepresentationConfig.from_dict({"experiment_name": "x", "phased": {}})

    def test_load_with_overrides(self, tmp_path: Path) -> None:
        path = tmp_path / "rep.yaml"
        path.write_text("experiment_name: t\nseed: 1\n", encoding="utf-8")
        config = load_representation_config(path, overrides=["n_figure_samples=0"])
        assert config.n_figure_samples == 0
