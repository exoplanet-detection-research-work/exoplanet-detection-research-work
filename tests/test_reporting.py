"""Tests for candidate report generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from exodet.config.schema import ExperimentConfig
from exodet.inference.config import ReportStageConfig
from exodet.inference.containers import ScientificInferenceBatch, ScientificInferenceResult
from exodet.inference.false_positive import FalsePositiveAnalyzer
from exodet.inference.parameter_fit import fit_transit_parameters
from exodet.inference.physical import estimate_physical_parameters
from exodet.reporting.report import ReportGenerator
from tests.ml_fixtures import make_labeled_sample, make_representation_dataset


def _make_result(sample: object) -> ScientificInferenceResult:
    transit = fit_transit_parameters(sample)
    physical = estimate_physical_parameters(sample, transit)
    fp = FalsePositiveAnalyzer().analyze(sample)
    return ScientificInferenceResult(
        sample_id=sample.sample_id,
        target_id=sample.target_id,
        candidate_id=sample.candidate.candidate_id,
        probability=0.85,
        classification="planet",
        confidence=0.85,
        transit=transit,
        physical=physical,
        false_positive=fp,
    )


class TestReportGeneration:
    def test_json_pdf_csv_outputs(self, tmp_path: Path) -> None:
        sample = make_labeled_sample(seed=0, n_global=32, n_local=32, n_features=8)
        result = _make_result(sample)
        cfg = ReportStageConfig(
            enabled=True,
            formats=("json", "pdf", "csv"),
            figure_dpi=80,
        )
        generator = ReportGenerator(cfg)
        report = generator.generate(result, sample, tmp_path)
        assert (tmp_path / f"{sample.sample_id}_report.json").is_file()
        assert (tmp_path / f"{sample.sample_id}_report.pdf").is_file()
        assert (tmp_path / f"{sample.sample_id}_summary.csv").is_file()
        assert "overview" in report.figure_paths

    def test_report_runner(self, tmp_path: Path) -> None:
        from exodet.reporting.runner import run_report_generation

        dataset = make_representation_dataset(n_samples=4, seed=1)
        batch = ScientificInferenceBatch(
            results=tuple(_make_result(s) for s in dataset.samples)
        )
        cfg_path = tmp_path / "report.yaml"
        cfg_path.write_text(
            "experiment_name: rep\n"
            "data: {source: {name: kepler_koi_archive, params: {}}, "
            "dataset: {name: kepler_lightcurves, params: {}}, "
            "train_fraction: 0.7, val_fraction: 0.15, stratify: true}\n"
            "model: {architecture: {name: linear_probe, params: {}}, features: []}\n"
            "training: {trainer: {name: supervised, params: {}}, epochs: 1, "
            "batch_size: 4, learning_rate: 0.001, early_stopping_patience: 0}\n"
            "evaluation: {metrics: [], decision_threshold: 0.5}\n"
            f"report: {{enabled: true, formats: [json], output_dir: '{tmp_path / 'reports'}'}}\n",
            encoding="utf-8",
        )
        out = run_report_generation(cfg_path, inference_batch=batch, dataset=dataset)
        assert out.is_dir()
