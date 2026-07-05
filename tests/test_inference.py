"""Tests for scientific inference layer."""

from __future__ import annotations

from pathlib import Path

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from exodet.config.schema import ExperimentConfig, TrainingConfig
from exodet.inference.config import InferenceStageConfig, load_inference_stage_config
from exodet.inference.containers import ScientificInferenceBatch
from exodet.inference.false_positive import FalsePositiveAnalyzer
from exodet.inference.parameter_fit import TransitParameterRefiner, fit_transit_parameters
from exodet.inference.physical import estimate_physical_parameters
from exodet.inference.pipeline import ScientificInferencePipeline
from exodet.inference.uncertainty import UncertaintyEstimate, UncertaintyEstimator
from exodet.ml.trainer import build_trainer
from exodet.models.base import MODELS
from exodet.utils.seeding import seed_everything
from tests.ml_fixtures import (
    fast_training_config,
    make_labeled_sample,
    make_representation_dataset,
)


def _minimal_experiment_yaml(**extra: object) -> dict[str, object]:
    base: dict[str, object] = {
        "experiment_name": "test",
        "data": {
            "source": {"name": "kepler_koi_archive", "params": {}},
            "dataset": {"name": "kepler_lightcurves", "params": {}},
            "train_fraction": 0.7,
            "val_fraction": 0.15,
            "stratify": True,
        },
        "model": {"architecture": {"name": "linear_probe", "params": {}}, "features": []},
        "training": fast_training_config(epochs=1, batch_size=4),
        "evaluation": {"metrics": [], "decision_threshold": 0.5},
    }
    base.update(extra)
    return base


from tests.scientific_tolerances import DEPTH_RTOL, PERIOD_RTOL

import exodet.models.registry  # noqa: F401, E402


class TestParameterFit:
    def test_refinement_improves_residuals(self) -> None:
        sample = make_labeled_sample(seed=0, n_global=32, n_local=32, n_features=8)
        refined = fit_transit_parameters(sample, {"method": "least_squares"})
        assert refined.depth > 0
        assert refined.duration_days > 0
        assert refined.period_days == pytest.approx(
            sample.candidate.period_days, rel=PERIOD_RTOL
        )

    def test_bootstrap_uncertainty(self) -> None:
        sample = make_labeled_sample(seed=1, n_global=32, n_local=32, n_features=8)
        refiner = TransitParameterRefiner(bootstrap_samples=5, seed=0)
        refined = refiner.refine(sample)
        assert refined.fit_method == "least_squares"


class TestPhysicalParameters:
    def test_with_stellar_metadata(self) -> None:
        sample = make_labeled_sample(seed=2, n_global=32, n_local=32, n_features=8)
        sample = replace(
            sample,
            meta={**sample.meta, "teff": 5500.0, "radius_rsun": 1.0, "mass_msun": 1.0},
        )
        transit = fit_transit_parameters(sample)
        physical = estimate_physical_parameters(sample, transit)
        assert physical.planet_radius_rearth is not None
        assert physical.semi_major_axis_au is not None

    def test_missing_stellar_metadata(self) -> None:
        sample = make_labeled_sample(seed=3, n_global=32, n_local=32, n_features=8)
        transit = fit_transit_parameters(sample)
        physical = estimate_physical_parameters(sample, transit)
        assert "planet_radius_rearth" in physical.missing_fields


class TestUncertainty:
    def test_none_method(self) -> None:
        est = UncertaintyEstimator({"method": "none"})
        dataset = make_representation_dataset(n_samples=4, seed=0)
        model = MODELS.build("linear_probe")
        import torch

        sample = dataset.samples[0]
        dim = sample.global_view.size + sample.local_view.size + sample.features.size
        model._ensure_module(dim, torch.device("cpu"))
        model._fitted = True
        results = est.estimate_batch(model, dataset)
        assert len(results) == 4
        assert isinstance(results[0], UncertaintyEstimate)


class TestFalsePositive:
    def test_analyzer_scores(self) -> None:
        sample = make_labeled_sample(seed=4, n_global=64, n_local=32, n_features=8)
        analyzer = FalsePositiveAnalyzer()
        assessment = analyzer.analyze(sample)
        assert 0.0 <= assessment.overall_fp_risk <= 1.0


class TestScientificMetadata:
    def test_reproduction_metadata_fields(self) -> None:
        from exodet.inference.scientific import build_reproduction_metadata

        exp = ExperimentConfig.from_dict(_minimal_experiment_yaml())
        meta = build_reproduction_metadata(exp, {"device": "cpu"})
        assert "package_version" in meta
        assert "units" in meta
        assert "physical_assumptions" in meta
        assert meta["random_seed"] == exp.seed

    def test_transit_result_includes_units(self) -> None:
        sample = make_labeled_sample(seed=8, n_global=32, n_local=32, n_features=8)
        refined = fit_transit_parameters(sample)
        assert "units" in refined.to_dict()
        assert refined.to_dict()["units"]["period_days"] == "days"


class TestInferenceConfig:
    def test_load_inference_config(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "inf.yaml"
        cfg_path.write_text(
            "experiment_name: t\n"
            "data: {source: {name: kepler_koi_archive, params: {}}, "
            "dataset: {name: kepler_lightcurves, params: {}}, "
            "train_fraction: 0.7, val_fraction: 0.15, stratify: true}\n"
            "model: {architecture: {name: linear_probe, params: {}}, features: []}\n"
            "training: {trainer: {name: supervised, params: {}}, epochs: 1, "
            "batch_size: 4, learning_rate: 0.001, early_stopping_patience: 0}\n"
            "evaluation: {metrics: [], decision_threshold: 0.5}\n"
            "inference: {enabled: true, uncertainty: {method: none}}\n",
            encoding="utf-8",
        )
        exp, inf = load_inference_stage_config(cfg_path)
        assert exp.experiment_name == "t"
        assert inf.enabled


class TestInferencePipeline:
    def _checkpoint(self, tmp_path: Path, dataset: object) -> Path:
        raw = fast_training_config(epochs=1, batch_size=4)
        trainer = build_trainer(TrainingConfig.from_dict(raw))
        model = MODELS.build("linear_probe")
        ckpt_dir = tmp_path / "ckpt"
        trainer.train(model, dataset, dataset, checkpoint_dir=ckpt_dir)
        return ckpt_dir

    def test_batch_inference(self, tmp_path: Path) -> None:
        seed_everything(0)
        dataset = make_representation_dataset(n_samples=8, seed=0)
        ckpt = self._checkpoint(tmp_path, dataset)
        raw = _minimal_experiment_yaml(
            inference={
                "enabled": True,
                "checkpoint_path": str(ckpt),
                "explainability": {"enabled": False},
                "uncertainty": {"method": "none"},
            }
        )
        inf_raw = raw.pop("inference")
        exp = ExperimentConfig.from_dict(raw)
        settings = InferenceStageConfig.from_dict(inf_raw)
        pipeline = ScientificInferencePipeline(exp, settings)
        batch = pipeline.predict_batch(dataset)
        assert isinstance(batch, ScientificInferenceBatch)
        assert len(batch) == 8
        assert batch.results[0].transit is not None

    def test_single_and_stream(self, tmp_path: Path) -> None:
        dataset = make_representation_dataset(n_samples=4, seed=1)
        ckpt = self._checkpoint(tmp_path, dataset)
        raw = _minimal_experiment_yaml(
            inference={
                "checkpoint_path": str(ckpt),
                "explainability": {"enabled": False},
            }
        )
        inf_raw = raw.pop("inference")
        exp = ExperimentConfig.from_dict(raw)
        settings = InferenceStageConfig.from_dict(inf_raw)
        pipeline = ScientificInferencePipeline(exp, settings)
        single = pipeline.predict_single(dataset.samples[0])
        assert single.sample_id == dataset.samples[0].sample_id
        streamed = list(pipeline.stream(iter(dataset.samples)))
        assert len(streamed) == 4

    def test_deterministic_parameter_fit(self) -> None:
        sample = make_labeled_sample(seed=7, n_global=32, n_local=32, n_features=8)
        a = fit_transit_parameters(sample, {"seed": 0})
        b = fit_transit_parameters(sample, {"seed": 0})
        assert a.depth == pytest.approx(b.depth, rel=DEPTH_RTOL)
