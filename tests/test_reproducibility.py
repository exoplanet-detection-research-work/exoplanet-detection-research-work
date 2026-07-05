"""Tests for reproducibility reporting."""

from __future__ import annotations

from pathlib import Path

import yaml

from exodet.benchmarking.config import load_benchmark_stage_config
from exodet.cli.main import main
from exodet.reproducibility.collector import checksum_file, collect_reproducibility_snapshot
from exodet.reproducibility.runner import run_reproducibility


def _minimal_config_dict(tmp_path: Path) -> dict:
    return {
        "experiment_name": "repro_test",
        "seed": 99,
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
            "trainer": {"name": "supervised", "params": {"backend": "sklearn"}},
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 1e-3,
            "early_stopping_patience": 0,
        },
        "evaluation": {"metrics": [], "decision_threshold": 0.5},
        "benchmark": {"enabled": False},
        "sensitivity": {"enabled": False},
    }


class TestReproducibility:
    def test_checksum_stable(self, tmp_path: Path) -> None:
        path = tmp_path / "file.bin"
        path.write_bytes(b"exodet")
        assert checksum_file(path) == checksum_file(path)
        assert len(checksum_file(path)) == 64

    def test_collect_snapshot(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        raw = _minimal_config_dict(tmp_path)
        config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        experiment, _, _, _, _ = load_benchmark_stage_config(config_path)
        snap = collect_reproducibility_snapshot(experiment, config_path=config_path)
        assert snap["random_seed"] == 99
        assert "package_versions" in snap
        assert snap["configuration_snapshot"]["config_checksum"]

    def test_run_reproducibility(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(_minimal_config_dict(tmp_path)), encoding="utf-8")
        payload = run_reproducibility(config_path)
        assert Path(payload["report_paths"]["json"]).is_file()
        assert Path(payload["report_paths"]["markdown"]).is_file()

    def test_reproducibility_cli(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(_minimal_config_dict(tmp_path)), encoding="utf-8")
        assert main(["reproducibility", "-c", str(config_path)]) == 0
