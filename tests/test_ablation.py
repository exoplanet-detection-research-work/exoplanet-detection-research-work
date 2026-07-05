"""Tests for ablation framework."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from exodet.ablation.runner import run_ablation
from exodet.cli.main import main
from tests.ml_fixtures import make_representation_dataset


@pytest.fixture()
def ablation_config(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "data" / "processed" / "dataset"
    dataset_root.mkdir(parents=True)
    data = make_representation_dataset(n_samples=36, n_stars=6, seed=1)
    samples = data.samples
    type(data)(samples[:24], version="test").save(dataset_root / "train.npz")
    type(data)(samples[24:30], version="test").save(dataset_root / "validation.npz")
    type(data)(samples[30:], version="test").save(dataset_root / "test.npz")

    config = {
        "experiment_name": "ablation_test",
        "seed": 1,
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
            "batch_size": 8,
            "learning_rate": 1e-3,
            "early_stopping_patience": 0,
        },
        "evaluation": {"metrics": [], "decision_threshold": 0.5},
        "ablation": {
            "enabled": True,
            "backend": "sklearn",
            "baseline_model": "logistic_regression",
            "variants": [
                {"id": "a", "architecture": "logistic_regression", "label": "LR"},
                {"id": "b", "architecture": "random_forest", "label": "RF"},
            ],
        },
        "benchmark": {"enabled": False},
        "sensitivity": {"enabled": False},
    }
    path = tmp_path / "ablation.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


class TestAblation:
    def test_run_ablation_table(self, ablation_config: Path) -> None:
        payload = run_ablation(ablation_config)
        assert len(payload["variants"]) == 2
        completed = [v for v in payload["variants"] if v["status"] == "completed"]
        assert len(completed) == 2
        assert "comparison_table" in payload

    def test_ablation_cli(self, ablation_config: Path) -> None:
        assert main(["ablation", "-c", str(ablation_config)]) == 0
