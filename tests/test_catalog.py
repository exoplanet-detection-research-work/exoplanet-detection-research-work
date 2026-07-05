"""Tests for catalog builder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from exodet.catalog.builder import CatalogBuilder
from exodet.catalog.runner import run_catalog_build
from exodet.inference.config import CatalogStageConfig
from exodet.inference.containers import ScientificInferenceBatch, ScientificInferenceResult
from exodet.inference.parameter_fit import fit_transit_parameters
from tests.ml_fixtures import make_labeled_sample


def _entry_result(sample: object, prob: float) -> ScientificInferenceResult:
    return ScientificInferenceResult(
        sample_id=sample.sample_id,
        target_id=sample.target_id,
        candidate_id=sample.candidate.candidate_id,
        probability=prob,
        classification="planet" if prob > 0.5 else "not_planet",
        confidence=prob,
        transit=fit_transit_parameters(sample),
    )


class TestCatalogBuilder:
    def test_build_and_export(self, tmp_path: Path) -> None:
        samples = [make_labeled_sample(seed=i, n_global=32, n_local=32, n_features=8) for i in range(3)]
        batch = ScientificInferenceBatch(
            results=tuple(_entry_result(s, 0.2 + 0.2 * i) for i, s in enumerate(samples))
        )
        builder = CatalogBuilder(
            CatalogStageConfig(formats=("csv", "json"), min_confidence=0.0)
        )
        entries = builder.build(batch)
        assert len(entries) == 3
        paths = builder.export(entries, tmp_path)
        assert "csv" in paths
        df = pd.read_csv(paths["csv"])
        assert "tic_id" in df.columns
        assert len(df) == 3

    def test_min_confidence_filter(self) -> None:
        sample = make_labeled_sample(seed=0, n_global=32, n_local=32, n_features=8)
        batch = ScientificInferenceBatch(results=(_entry_result(sample, 0.1),))
        builder = CatalogBuilder(CatalogStageConfig(min_confidence=0.5))
        assert builder.build(batch) == []

    def test_catalog_runner(self, tmp_path: Path) -> None:
        sample = make_labeled_sample(seed=2, n_global=32, n_local=32, n_features=8)
        batch = ScientificInferenceBatch(results=(_entry_result(sample, 0.9),))
        cfg_path = tmp_path / "cat.yaml"
        cfg_path.write_text(
            "experiment_name: cat\n"
            f"paths: {{report_dir: '{tmp_path / 'reports'}'}}\n"
            "data: {source: {name: kepler_koi_archive, params: {}}, "
            "dataset: {name: kepler_lightcurves, params: {}}, "
            "train_fraction: 0.7, val_fraction: 0.15, stratify: true}\n"
            "model: {architecture: {name: linear_probe, params: {}}, features: []}\n"
            "training: {trainer: {name: supervised, params: {}}, epochs: 1, "
            "batch_size: 4, learning_rate: 0.001, early_stopping_patience: 0}\n"
            "evaluation: {metrics: [], decision_threshold: 0.5}\n"
            "catalog: {enabled: true, formats: [csv, json]}\n",
            encoding="utf-8",
        )
        paths = run_catalog_build(cfg_path, inference_batch=batch)
        assert "csv" in paths
