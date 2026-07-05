"""Tests for experiment orchestration system."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from exodet.cli.main import main
from exodet.experiments.comparison import build_leaderboard
from exodet.experiments.config import load_experiments_stage_config
from exodet.experiments.database import ExperimentDatabase, ExperimentRecord
from exodet.experiments.manager import ExperimentManager
from exodet.experiments.performance import benchmark_database_scales
from exodet.experiments.recovery import verify_checkpoint
from exodet.experiments.sweeps import _iter_grid, _iter_random
from exodet.experiments.tables import write_latex_table, write_markdown_table
from exodet.experiments.templates import apply_template, list_templates
from exodet.experiments.validation import _compare_metrics
from tests.ml_fixtures import make_representation_dataset


def _experiment_config_dict(tmp_path: Path, **extra: object) -> dict:
    base: dict = {
        "experiment_name": "exp_test",
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
        "evaluation": {"metrics": [], "decision_threshold": 0.5},
        "experiments": {
            "enabled": True,
            "tags": ["test"],
            "stage": "train",
            "template": None,
        },
        "sweep": {"enabled": False},
        "artifacts": {"enabled": True, "organize": False},
        "reproduce": {"enabled": False},
        "benchmark": {"enabled": False},
        "sensitivity": {"enabled": False},
        "ablation": {"enabled": False},
    }
    base.update(extra)
    return base


@pytest.fixture()
def experiment_config(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "data" / "processed" / "dataset"
    dataset_root.mkdir(parents=True)
    data = make_representation_dataset(n_samples=40, n_stars=8, seed=0)
    samples = data.samples
    type(data)(samples[:28], version="test").save(dataset_root / "train.npz")
    type(data)(samples[28:34], version="test").save(dataset_root / "validation.npz")
    type(data)(samples[34:], version="test").save(dataset_root / "test.npz")

    config = _experiment_config_dict(tmp_path)
    path = tmp_path / "experiments.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


class TestExperimentDatabase:
    def test_register_and_search(self, tmp_path: Path) -> None:
        db = ExperimentDatabase(tmp_path / "index.json")
        rec = ExperimentRecord(
            experiment_id="abc123",
            name="test_exp",
            status="completed",
            tags=("baseline",),
            metrics={"roc_auc": 0.85},
        )
        db.register(rec)
        assert db.get("abc123") is not None
        found = db.search(tags=("baseline",), status="completed")
        assert len(found) == 1

    def test_bulk_insert_100_records(self, tmp_path: Path) -> None:
        db = ExperimentDatabase(tmp_path / "bench.json")
        records = [
            ExperimentRecord(
                experiment_id=f"e{i:04d}",
                name=f"exp_{i}",
                status="completed",
                metrics={"roc_auc": 0.5 + i / 200.0},
            )
            for i in range(100)
        ]
        db.bulk_insert(records)
        assert db.size == 100


class TestTemplates:
    def test_list_templates(self) -> None:
        templates = list_templates()
        assert "sklearn_baseline" in templates
        assert "hybrid_model" in templates

    def test_apply_template(self) -> None:
        merged = apply_template("sklearn_baseline", {"experiment_name": "custom"})
        assert merged["model"]["architecture"]["name"] == "logistic_regression"


class TestSweeps:
    def test_grid_iterator(self) -> None:
        params = {"C": [0.1, 1.0], "max_iter": [100, 200]}
        combos = list(_iter_grid(params, max_trials=0))
        assert len(combos) == 4

    def test_random_iterator(self) -> None:
        combos = list(_iter_random({"C": [0.1, 1.0, 10.0]}, 5, seed=0))
        assert len(combos) == 5


class TestComparison:
    def test_leaderboard_ranking(self) -> None:
        records = [
            ExperimentRecord("a", "a", metrics={"roc_auc": 0.7}),
            ExperimentRecord("b", "b", metrics={"roc_auc": 0.9}),
            ExperimentRecord("c", "c", metrics={"roc_auc": 0.8}),
        ]
        lb = build_leaderboard(records, "roc_auc")
        assert lb.rows[0]["experiment_id"] == "b"
        assert lb.rows[0]["rank"] == 1


class TestTables:
    def test_markdown_and_latex(self) -> None:
        rows = [{"rank": 1, "name": "exp", "metric_value": 0.9}]
        md = write_markdown_table(rows, ["rank", "name", "metric_value"])
        assert "0.9" in md
        tex = write_latex_table(rows, ["rank", "name", "metric_value"])
        assert "\\begin{table}" in tex


class TestValidation:
    def test_metric_comparison(self) -> None:
        ok, deltas = _compare_metrics({"roc_auc": 0.9}, {"roc_auc": 0.90001}, 1e-3)
        assert ok
        assert deltas["roc_auc"] < 1e-3


class TestRecovery:
    def test_verify_missing_checkpoint(self, tmp_path: Path) -> None:
        result = verify_checkpoint(tmp_path / "missing.pt")
        assert not result.valid


class TestPerformance:
    def test_database_benchmark_scales(self, tmp_path: Path) -> None:
        results = benchmark_database_scales(tmp_path, scales=(100,))
        assert len(results) == 1
        assert results[0].n_records == 100
        assert results[0].insert_seconds >= 0


class TestExperimentRunner:
    def test_run_experiment(self, experiment_config: Path) -> None:
        from exodet.experiments.runner import run_experiment

        payload = run_experiment(experiment_config)
        assert "experiment_id" in payload
        eid = payload["experiment_id"]
        out_dir = Path(payload["record"]["output_dir"])
        assert (out_dir / "experiment.json").is_file()
        assert (out_dir / "run_summary.json").is_file()

    def test_run_sweep(self, experiment_config: Path) -> None:
        raw = yaml.safe_load(experiment_config.read_text())
        raw["sweep"] = {
            "enabled": True,
            "method": "grid",
            "parameters": {"C": [0.1, 1.0]},
            "max_trials": 2,
            "model_name": "logistic_regression",
        }
        sweep_config = experiment_config.parent / "sweep.yaml"
        sweep_config.write_text(yaml.safe_dump(raw), encoding="utf-8")
        from exodet.experiments.runner import run_experiment_sweep

        payload = run_experiment_sweep(sweep_config)
        assert "sweep_id" in payload
        assert len(payload["result"]["trials"]) >= 1

    def test_leaderboard(self, experiment_config: Path) -> None:
        from exodet.experiments.runner import run_experiment, run_leaderboard

        run_experiment(experiment_config)
        payload = run_leaderboard(experiment_config)
        assert "comparison" in payload

    def test_experiment_manager_register(self, experiment_config: Path) -> None:
        experiment, stage, _, _, _, _ = load_experiments_stage_config(experiment_config)
        db = ExperimentDatabase(experiment_config.parent / "index.json")
        manager = ExperimentManager(
            experiment, database=db, stage_config=stage, config_path=experiment_config
        )
        rec = manager.register()
        assert rec.experiment_id
        assert Path(rec.output_dir).is_dir()


class TestExperimentCli:
    def test_experiment_cli(self, experiment_config: Path) -> None:
        assert main(["experiment", "-c", str(experiment_config)]) == 0

    def test_sweep_cli(self, experiment_config: Path) -> None:
        raw = yaml.safe_load(experiment_config.read_text())
        raw["sweep"] = {
            "enabled": True,
            "method": "random",
            "parameters": {"C": [0.1, 1.0]},
            "random_samples": 2,
        }
        path = experiment_config.parent / "sweep_cli.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        assert main(["sweep", "-c", str(path)]) == 0

    def test_leaderboard_cli(self, experiment_config: Path) -> None:
        main(["experiment", "-c", str(experiment_config)])
        assert main(["leaderboard", "-c", str(experiment_config)]) == 0
