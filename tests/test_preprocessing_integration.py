"""Integration tests: full pipeline, runner, CLI, serialization, figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from exodet.cli.main import main
from exodet.config import load_config
from exodet.config.schema import PreprocessingConfig
from exodet.data.base import LightCurve
from exodet.data.serialization import load_light_curve, save_light_curve
from exodet.preprocessing import PreprocessingPipeline, run_preprocessing
from exodet.visualization.preprocessing import generate_preprocessing_figures

FULL_PIPELINE_STEPS = [
    {"name": "quality_filter", "params": {"bitmask": "default"}},
    {"name": "nan_removal", "params": {"strategy": "drop"}},
    {"name": "sector_stitch", "params": {}},
    {"name": "gap_detect", "params": {"factor": 5.0}},
    {"name": "gap_interpolate", "params": {"method": "linear", "max_gap_days": 0.5}},
    {
        "name": "wotan_detrend",
        "params": {"method": "biweight", "window_length_days": 0.5},
    },
    {"name": "sigma_clip", "params": {"sigma": 5.0, "clip_lower": False}},
    {"name": "normalize", "params": {"method": "median"}},
    {"name": "quality_metrics", "params": {"cdpp_duration_hours": 1.0}},
]


def _experiment_yaml(tmp_path: Path) -> str:
    steps = "\n".join(
        f"    - name: {step['name']}\n      params: {step['params']}"
        for step in FULL_PIPELINE_STEPS
    )
    return f"""\
experiment_name: integration_test
seed: 42
paths:
  data_dir: {tmp_path / 'data'}
  raw_dir: {tmp_path / 'data/raw'}
  interim_dir: {tmp_path / 'data/interim'}
  processed_dir: {tmp_path / 'data/processed'}
  output_dir: {tmp_path / 'outputs'}
  checkpoint_dir: {tmp_path / 'outputs/checkpoints'}
  figure_dir: {tmp_path / 'outputs/figures'}
  log_dir: {tmp_path / 'outputs/logs'}
  report_dir: {tmp_path / 'outputs/reports'}
logging:
  level: INFO
  to_file: false
data:
  source:
    name: dummy_source
  dataset:
    name: synthetic_tess
    params:
      n_targets: 2
      n_per_sector: 800
model:
  architecture:
    name: dummy_model
training:
  trainer:
    name: dummy_trainer
preprocessing:
  steps:
{steps}
"""


class TestFullPipeline:
    @pytest.fixture()
    def processed(self, tess_curve: LightCurve) -> LightCurve:
        config = PreprocessingConfig.from_dict({"steps": FULL_PIPELINE_STEPS})
        return PreprocessingPipeline.from_config(config).apply(tess_curve)

    def test_provenance_records_every_stage(
        self, processed: LightCurve, tess_curve: LightCurve
    ) -> None:
        assert len(processed.history) == len(FULL_PIPELINE_STEPS)
        assert tess_curve.history == []

    def test_output_is_clean(self, processed: LightCurve) -> None:
        assert np.isfinite(processed.flux).all()
        assert np.isfinite(processed.time).all()
        assert np.all(np.diff(processed.time) > 0)
        assert np.median(processed.flux) == pytest.approx(1.0, abs=1e-3)

    def test_transit_signal_preserved(self, processed: LightCurve) -> None:
        # Injected depth is 0.008; detrending must not erase it.
        assert processed.flux.min() < 1.0 - 0.003

    def test_metadata_artifacts_present(self, processed: LightCurve) -> None:
        assert processed.meta["sector_medians"]
        assert isinstance(processed.meta["gaps"], list)
        assert processed.meta["interpolated"].dtype == bool
        assert processed.meta["normalization"]["method"] == "median"
        assert processed.meta["quality_metrics"]["duty_cycle"] > 0
        assert processed.label == 1
        assert processed.mission == "tess"

    def test_defective_cadences_are_gone(self, processed: LightCurve) -> None:
        assert (processed.meta["quality"] & 128).sum() == 0


class TestSerializationRoundTrip:
    def test_round_trip_preserves_everything(
        self, tess_curve: LightCurve, tmp_path: Path
    ) -> None:
        config = PreprocessingConfig.from_dict({"steps": FULL_PIPELINE_STEPS})
        processed = PreprocessingPipeline.from_config(config).apply(tess_curve)
        path = save_light_curve(processed, tmp_path / "curve.npz")
        loaded = load_light_curve(path)

        np.testing.assert_array_equal(loaded.time, processed.time)
        np.testing.assert_array_equal(loaded.flux, processed.flux)
        np.testing.assert_array_equal(loaded.flux_err, processed.flux_err)
        assert loaded.target_id == processed.target_id
        assert loaded.label == processed.label
        assert loaded.mission == processed.mission
        assert loaded.history == processed.history
        np.testing.assert_array_equal(
            loaded.meta["interpolated"], processed.meta["interpolated"]
        )
        assert loaded.meta["quality_metrics"] == processed.meta["quality_metrics"]


class TestRunner:
    def test_runner_end_to_end(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(_experiment_yaml(tmp_path), encoding="utf-8")
        config = load_config(config_path)

        outputs = run_preprocessing(config)

        assert len(outputs) == 2
        assert all(path.is_file() for path in outputs)
        report = tmp_path / "outputs/reports/preprocessing_summary.json"
        assert report.is_file()
        loaded = load_light_curve(outputs[0])
        assert len(loaded.history) == len(FULL_PIPELINE_STEPS)

    def test_runner_writes_figures_for_first_target(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(_experiment_yaml(tmp_path), encoding="utf-8")
        run_preprocessing(load_config(config_path))

        figure_dir = tmp_path / "outputs/figures"
        stems = {path.name for path in figure_dir.iterdir()}
        for kind in ("raw", "detrended", "clipped", "normalization"):
            assert f"tic_1000_{kind}.pdf" in stems
            assert f"tic_1000_{kind}.png" in stems


class TestCli:
    def test_cli_preprocess_end_to_end(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(_experiment_yaml(tmp_path), encoding="utf-8")

        assert main(["preprocess", "-c", str(config_path)]) == 0
        assert "Preprocessed 2 target(s)" in capsys.readouterr().out
        assert (tmp_path / "data/processed/tic_1000.npz").is_file()
        assert (tmp_path / "data/processed/tic_1001.npz").is_file()

    def test_cli_preprocess_unknown_dataset_fails_cleanly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        yaml_text = _experiment_yaml(tmp_path).replace(
            "name: synthetic_tess", "name: does_not_exist"
        )
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_text, encoding="utf-8")

        assert main(["preprocess", "-c", str(config_path)]) == 1
        assert "does_not_exist" in capsys.readouterr().err


class TestFigures:
    def test_generate_figures_without_optional_metadata(
        self, clean_tess_curve: LightCurve, tmp_path: Path
    ) -> None:
        # A pipeline without detrending/clipping still yields raw and
        # normalization comparison figures.
        written = generate_preprocessing_figures(
            clean_tess_curve, clean_tess_curve, tmp_path
        )
        names = {path.name for path in written}
        assert "tic_123456789_raw.png" in names
        assert "tic_123456789_normalization.png" in names
        assert not any("detrended" in name for name in names)
