"""Tests for benchmarking suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from exodet.benchmarking.reports import BenchmarkReport, write_benchmark_reports
from exodet.benchmarking.runner import run_benchmark, run_sensitivity
from exodet.benchmarking.sensitivity import apply_perturbation
from exodet.benchmarking.statistics import (
    bootstrap_confidence_interval,
    compare_model_predictions,
    mcnemar_test,
    paired_t_test,
    wilcoxon_signed_rank_test,
)
from exodet.cli.main import main
from tests.ml_fixtures import make_representation_dataset


@pytest.fixture()
def benchmark_config(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "data" / "processed" / "dataset"
    dataset_root.mkdir(parents=True)
    data = make_representation_dataset(n_samples=48, n_stars=8, seed=0)
    samples = data.samples
    train = type(data)(samples[:32], version="test")
    val = type(data)(samples[32:40], version="test")
    test = type(data)(samples[40:], version="test")
    train.save(dataset_root / "train.npz")
    val.save(dataset_root / "validation.npz")
    test.save(dataset_root / "test.npz")

    config = {
        "experiment_name": "benchmark_test",
        "seed": 0,
        "paths": {
            "data_dir": str(tmp_path / "data"),
            "raw_dir": str(tmp_path / "data" / "raw"),
            "interim_dir": str(tmp_path / "data" / "interim"),
            "processed_dir": str(tmp_path / "data" / "processed"),
            "output_dir": str(tmp_path / "outputs"),
            "checkpoint_dir": str(tmp_path / "outputs" / "checkpoints"),
            "figure_dir": str(tmp_path / "outputs" / "figures"),
            "log_dir": str(tmp_path / "outputs" / "logs"),
            "report_dir": str(tmp_path / "outputs" / "reports"),
        },
        "data": {
            "source": {"name": "dummy", "params": {}},
            "dataset": {"name": "dummy", "params": {}},
            "train_fraction": 0.7,
            "val_fraction": 0.15,
            "stratify": True,
        },
        "model": {"architecture": {"name": "logistic_regression", "params": {}}, "features": []},
        "training": {
            "trainer": {
                "name": "supervised",
                "params": {"backend": "sklearn", "use_views": "both"},
            },
            "epochs": 1,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "early_stopping_patience": 0,
        },
        "evaluation": {
            "metrics": [{"name": "accuracy", "params": {}}],
            "decision_threshold": 0.5,
        },
        "benchmark": {
            "enabled": True,
            "models": ["logistic_regression", "random_forest"],
            "reports": {"formats": ["json", "markdown", "csv"]},
            "statistics": {"n_bootstrap": 100},
            "calibration": {"enabled": True, "n_bins": 5},
            "error_analysis": {"enabled": True},
            "cross_mission": {"enabled": True},
        },
        "sensitivity": {
            "enabled": True,
            "perturbations": ["gaussian_noise"],
            "levels": [0.0, 0.05],
        },
        "hyperparameter": {"enabled": False},
    }
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


class TestStatistics:
    def test_mcnemar_symmetric(self) -> None:
        y = np.array([1, 0, 1, 0], dtype=np.int_)
        a = np.array([1, 0, 0, 0], dtype=np.int_)
        b = np.array([0, 0, 1, 1], dtype=np.int_)
        result = mcnemar_test(y, a, b)
        assert result.n01 == 1
        assert result.n10 == 2
        assert 0.0 <= result.p_value <= 1.0

    def test_bootstrap_ci_contains_mean(self) -> None:
        values = np.linspace(0.2, 0.8, 50)
        ci = bootstrap_confidence_interval(values, n_bootstrap=200, seed=0)
        assert ci.lower <= ci.point_estimate <= ci.upper

    def test_paired_tests(self) -> None:
        a = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float64)
        b = np.array([0.5, 0.4, 0.3, 0.2], dtype=np.float64)
        t_result = paired_t_test(a, b)
        w_result = wilcoxon_signed_rank_test(a, b)
        assert t_result.p_value < 0.05
        assert w_result.p_value <= 1.0

    def test_compare_model_predictions(self) -> None:
        y = np.array([1, 0, 1, 0, 1, 0], dtype=np.int_)
        preds = {
            "m1": np.array([1, 0, 1, 0, 0, 0], dtype=np.int_),
            "m2": np.array([1, 0, 0, 0, 1, 1], dtype=np.int_),
        }
        probs = {
            "m1": np.array([0.9, 0.1, 0.8, 0.2, 0.4, 0.3]),
            "m2": np.array([0.7, 0.2, 0.3, 0.1, 0.9, 0.8]),
        }
        out = compare_model_predictions(y, preds, probs, n_bootstrap=50, seed=0)
        assert "pairwise" in out
        assert "bootstrap_accuracy" in out


class TestSensitivity:
    def test_gaussian_noise_changes_features(self) -> None:
        features = np.ones((4, 8), dtype=np.float64)
        labels = np.array([0, 1, 0, 1], dtype=np.int_)
        result = apply_perturbation(features, labels, "gaussian_noise", 0.1, seed=0)
        assert not np.allclose(result.features, features)


class TestBenchmarkRunner:
    def test_run_benchmark_outputs(self, benchmark_config: Path, tmp_path: Path) -> None:
        report = run_benchmark(benchmark_config)
        output_root = tmp_path / "outputs" / "reports" / "benchmark"
        # runner uses report_dir/benchmark by default
        output_root = Path(yaml.safe_load(benchmark_config.read_text())["paths"]["report_dir"]) / "benchmark"
        assert (output_root / "benchmark_manifest.json").is_file()
        assert report.experiment_name == "benchmark_test"

    def test_run_sensitivity(self, benchmark_config: Path) -> None:
        payload = run_sensitivity(benchmark_config)
        assert "gaussian_noise" in payload["curves"]

    def test_report_generation(self, tmp_path: Path) -> None:
        report = BenchmarkReport(
            experiment_name="unit",
            dataset_summary={"test": {"n_samples": 10}},
            training_configuration={"backend": "sklearn"},
            model_results=[{"name": "lr", "metrics": {"accuracy": 0.8}, "runtime_seconds": 1.0}],
            conclusions=["ok"],
        )
        paths = write_benchmark_reports(report, tmp_path, formats=("json", "markdown", "html", "csv"))
        assert Path(paths["json"]).is_file()
        assert Path(paths["markdown"]).is_file()


class TestBenchmarkCli:
    def test_benchmark_cli(self, benchmark_config: Path) -> None:
        assert main(["benchmark", "-c", str(benchmark_config)]) == 0

    def test_sensitivity_cli(self, benchmark_config: Path) -> None:
        assert main(["sensitivity", "-c", str(benchmark_config)]) == 0
