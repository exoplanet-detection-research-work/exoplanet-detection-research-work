"""Tests for incremental update pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from exodet.cli.main import main
from exodet.config.schema import ExperimentConfig
from exodet.data.serialization import save_light_curve
from exodet.ml.checkpoints import CheckpointManager
from exodet.representation.config import RepresentationConfig
from exodet.representation.containers import RepresentationDataset
from exodet.tce import inject_box_transit, make_noise_light_curve
from exodet.tce.candidate import save_candidates
from exodet.tce.config import TCESearchConfig
from exodet.update.checkpoint_manager import (
    discover_checkpoint,
    list_experiment_checkpoints,
)
from exodet.update.config import UpdateStageConfig, load_update_stage_config
from exodet.update.dataset_registry import DatasetRegistry, TargetRecord
from exodet.update.resume import apply_training_resume, checkpoint_extra_state
from exodet.update.update_pipeline import (
    TargetStageState,
    UpdatePipeline,
    merge_tce_catalog,
    parse_tic_ids_from_file,
    resolve_update_inputs,
)
from exodet.update.versioning import append_to_splits, load_or_create_manifest
from tests.conftest import make_synthetic_tess_curve
from tests.ml_fixtures import (
    make_labeled_sample,
)
from tests.test_representation_integration import fast_representation_config
from tests.test_tce_integration import fast_tce_config


def _paths_dict(tmp_path: Path) -> dict[str, str]:
    return {
        "data_dir": str(tmp_path / "data"),
        "raw_dir": str(tmp_path / "data" / "raw"),
        "interim_dir": str(tmp_path / "data" / "interim"),
        "processed_dir": str(tmp_path / "data" / "processed"),
        "output_dir": str(tmp_path / "outputs"),
        "checkpoint_dir": str(tmp_path / "outputs" / "checkpoints"),
        "figure_dir": str(tmp_path / "outputs" / "figures"),
        "log_dir": str(tmp_path / "outputs" / "logs"),
        "report_dir": str(tmp_path / "outputs" / "reports"),
    }


def _experiment_config(tmp_path: Path) -> ExperimentConfig:
    raw = {
        "experiment_name": "update_test",
        "seed": 0,
        "paths": _paths_dict(tmp_path),
        "logging": {"level": "INFO", "to_file": False},
        "data": {
            "source": {"name": "synthetic_tic", "params": {}},
            "dataset": {"name": "synthetic_tess", "params": {"n_targets": 1}},
            "train_fraction": 0.7,
            "val_fraction": 0.15,
            "stratify": True,
        },
        "preprocessing": {
            "steps": [
                {"name": "nan_removal", "params": {"strategy": "drop"}},
                {"name": "normalize", "params": {"method": "median"}},
            ]
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
    }
    return ExperimentConfig.from_dict(raw)


def _injected_target(target_id: str = "TIC 999001", seed: int = 0):
    injected = inject_box_transit(
        make_noise_light_curve(
            target_id=target_id,
            n_points=12_000,
            noise_level=5e-4,
            seed=seed,
        ),
        period_days=2.1,
        duration_days=0.1,
        depth=0.005,
        epoch_days=0.5,
    )
    return injected


class TestInputParsing:
    def test_parse_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "tics.csv"
        path.write_text("tic_id\n123\n456\n", encoding="utf-8")
        assert parse_tic_ids_from_file(path) == ["123", "456"]

    def test_parse_txt(self, tmp_path: Path) -> None:
        path = tmp_path / "tics.txt"
        path.write_text("# comment\n789\n", encoding="utf-8")
        assert parse_tic_ids_from_file(path) == ["789"]

    def test_parse_json(self, tmp_path: Path) -> None:
        path = tmp_path / "tics.json"
        path.write_text('{"tic_ids": ["111", "222"]}', encoding="utf-8")
        assert parse_tic_ids_from_file(path) == ["111", "222"]

    def test_resolve_inputs_from_processed_dir(self, tmp_path: Path) -> None:
        processed = tmp_path / "processed"
        processed.mkdir()
        curve = _injected_target()
        save_light_curve(curve, processed / "tic_999001.npz")
        update = UpdateStageConfig(processed_dir=str(processed))
        inputs = resolve_update_inputs(update)
        assert inputs.source == "files"
        assert len(inputs.curves) == 1


class TestDatasetRegistry:
    def test_skip_duplicate_tic(self, tmp_path: Path) -> None:
        registry = DatasetRegistry(tmp_path / "dataset_registry.json")
        registry.register(
            TargetRecord(
                tic_id="123456789",
                target_id="TIC 123456789",
                mission="TESS",
                download_date="2026-01-01T00:00:00+00:00",
            )
        )
        assert registry.should_process("123456789", force=False) is False
        assert registry.should_process("123456789", force=True) is True


class TestVersioning:
    def test_append_without_duplicates(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        manifest_path = dataset_dir / "manifest.json"
        existing = make_labeled_sample(seed=1, target_id="TIC 9001")
        RepresentationDataset([existing], version="v1").save(dataset_dir / "train.npz")
        new_sample = make_labeled_sample(seed=2, target_id="TIC 9002")
        duplicate = make_labeled_sample(seed=1, target_id="TIC 9001")
        info = append_to_splits(
            dataset_dir,
            [new_sample, duplicate],
            split="train",
            version="v1",
            experiment_name="exp",
            manifest_path=manifest_path,
        )
        assert info["n_added"] == 1
        loaded = RepresentationDataset.load(dataset_dir / "train.npz")
        assert len(loaded) == 2
        manifest = load_or_create_manifest(
            manifest_path, version="v1", experiment_name="exp"
        )
        assert manifest.n_samples["train"] == 2


class TestScalerReload:
    def test_load_existing_scaler_on_append(self, tmp_path: Path) -> None:
        from exodet.representation.scaling import FeatureScaler

        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        names = tuple(f"f{i}" for i in range(4))
        matrix = np.stack(
            [make_labeled_sample(seed=i, n_features=4).features for i in range(3)]
        )
        scaler = FeatureScaler(method="robust", log_features=())
        scaler.fit(matrix, names)
        scaler.save(dataset_dir / "feature_scaler.json")

        pipeline = UpdatePipeline(
            _experiment_config(tmp_path),
            UpdateStageConfig(),
            fast_tce_config(),
            fast_representation_config(),
        )
        samples = [make_labeled_sample(seed=10, n_features=4)]
        scaled = pipeline._scale_samples(samples, dataset_dir)
        assert len(scaled) == 1
        assert scaled[0].features.shape == samples[0].features.shape


class TestCheckpointManager:
    def test_discover_best_and_latest(self, tmp_path: Path) -> None:
        ckpt_root = tmp_path / "checkpoints"
        exp_dir = ckpt_root / "update_test"
        exp_dir.mkdir(parents=True)
        manager = CheckpointManager(directory=exp_dir, save_best=True, save_last=True)
        manager.save(
            epoch=1,
            model_state={"w": torch.tensor([1.0])},
            metrics={"val_loss": 0.4},
        )
        manager.save(
            epoch=2,
            model_state={"w": torch.tensor([2.0])},
            metrics={"val_loss": 0.2},
        )
        listed = list_experiment_checkpoints(ckpt_root)
        assert len(listed) == 1
        best = discover_checkpoint(ckpt_root, "update_test", selection="best")
        assert best.path.name == "best.pt"
        latest = discover_checkpoint(ckpt_root, "update_test", selection="latest")
        assert latest.path.is_file()


class TestResume:
    def test_apply_training_resume_sets_path(self, tmp_path: Path) -> None:
        experiment = _experiment_config(tmp_path)
        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"")
        from exodet.update.checkpoint_manager import CheckpointDiscovery

        discovery = CheckpointDiscovery(
            path=ckpt,
            selection="explicit",
            experiment_name="update_test",
            epoch=1,
            metrics={"val_loss": 0.1},
        )
        updated = apply_training_resume(experiment, discovery)
        assert updated.training.trainer.params["resume_from"] == str(ckpt)

    def test_checkpoint_extra_state(self, tmp_path: Path) -> None:
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        manager = CheckpointManager(directory=ckpt_dir, save_last=True)
        manager.save(
            epoch=3,
            model_state={"w": torch.tensor([1.0])},
            optimizer_state={"state": {}},
            metrics={"val_loss": 0.3},
            extra={"global_step": 42, "ema": {"decay": 0.999}},
        )
        state = checkpoint_extra_state(ckpt_dir / "last.pt")
        assert state["epoch"] == 3
        assert state["has_optimizer"] is True
        assert state["extra"]["global_step"] == 42


class TestStageRecovery:
    def test_resume_from_last_completed_stage(self, tmp_path: Path) -> None:
        state = TargetStageState(tic_id="999001", target_id="TIC 999001")
        state.mark_complete("download")
        state.mark_complete("preprocess")
        assert state.is_complete("preprocess")
        assert not state.is_complete("tce")


class TestUpdatePipeline:
    def test_end_to_end_processed_input(self, tmp_path: Path) -> None:
        experiment = _experiment_config(tmp_path)
        for path in (
            experiment.paths.processed_dir,
            experiment.paths.report_dir,
            experiment.paths.raw_dir,
            experiment.paths.interim_dir,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)

        curve = _injected_target()
        processed = Path(experiment.paths.processed_dir)
        save_light_curve(curve, processed / "tic_999001.npz")
        save_candidates([], Path(experiment.paths.report_dir) / "tce_candidates.json")

        update = UpdateStageConfig(
            force_reprocess=False,
            append_split="train",
            download={"backend": "synthetic"},
        )
        pipeline = UpdatePipeline(
            experiment,
            update,
            fast_tce_config(
                paths={"report_dir": experiment.paths.report_dir, "processed_dir": experiment.paths.processed_dir},
                n_figure_targets=0,
            ),
            fast_representation_config(
                paths={
                    "processed_dir": experiment.paths.processed_dir,
                    "report_dir": experiment.paths.report_dir,
                    "interim_dir": experiment.paths.interim_dir,
                },
                n_figure_samples=0,
            ),
        )
        from exodet.update.update_pipeline import UpdateInputs

        summary = pipeline.run(
            UpdateInputs(curves=(curve,), source="files"),
        )
        assert summary["n_success"] == 1
        registry = DatasetRegistry(processed / "dataset_registry.json")
        assert registry.contains("999001")
        assert (processed / "dataset" / "train.npz").is_file()

    def test_duplicate_target_skipped(self, tmp_path: Path) -> None:
        experiment = _experiment_config(tmp_path)
        processed = Path(experiment.paths.processed_dir)
        processed.mkdir(parents=True)
        registry_path = processed / "dataset_registry.json"
        registry = DatasetRegistry(registry_path)
        registry.register(
            TargetRecord(
                tic_id="999001",
                target_id="TIC 999001",
                mission="TESS",
                download_date="2026-01-01T00:00:00+00:00",
            )
        )
        curve = _injected_target()
        update = UpdateStageConfig(force_reprocess=False)
        pipeline = UpdatePipeline(
            experiment,
            update,
            fast_tce_config(),
            fast_representation_config(),
        )
        from exodet.update.update_pipeline import UpdateInputs

        summary = pipeline.run(UpdateInputs(curves=(curve,), source="files"))
        assert summary["n_skipped"] == 1

    def test_merge_tce_catalog(self, tmp_path: Path) -> None:
        from tests.test_tce import make_candidate

        catalog = tmp_path / "tce_candidates.json"
        first = make_candidate(candidate_id="A-01", target_id="TIC 1")
        save_candidates([first], catalog)
        second = make_candidate(candidate_id="B-01", target_id="TIC 2")
        merged = merge_tce_catalog(catalog, [second])
        assert len(merged) == 2


class TestConfigLoader:
    def test_load_update_stage_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "update.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "experiment_name": "update_cfg",
                    "seed": 0,
                    "paths": _paths_dict(tmp_path),
                    "logging": {"to_file": False},
                    "data": {
                        "source": {"name": "synthetic_tic", "params": {}},
                        "dataset": {"name": "synthetic_tess", "params": {}},
                        "train_fraction": 0.7,
                        "val_fraction": 0.15,
                        "stratify": True,
                    },
                    "preprocessing": {"steps": []},
                    "model": {
                        "architecture": {"name": "logistic_regression", "params": {}},
                        "features": [],
                    },
                    "training": {
                        "trainer": {"name": "supervised", "params": {}},
                        "epochs": 1,
                        "batch_size": 8,
                        "learning_rate": 1e-3,
                        "early_stopping_patience": 0,
                    },
                    "evaluation": {"metrics": [], "decision_threshold": 0.5},
                    "grid": {"name": "bls_auto", "params": {}},
                    "folding": {"name": "standard", "params": {}},
                    "global_view": {"name": "global", "params": {"n_bins": 101}},
                    "local_view": {"name": "local", "params": {"n_bins": 41}},
                    "splitting": {
                        "name": "star",
                        "params": {"validation_fraction": 0.0, "test_fraction": 0.0},
                    },
                    "update": {"enabled": True, "resume_training": False},
                }
            ),
            encoding="utf-8",
        )
        experiment, update, tce, rep, _ = load_update_stage_config(config_path)
        assert experiment.experiment_name == "update_cfg"
        assert update.enabled is True
        assert isinstance(tce, TCESearchConfig)
        assert isinstance(rep, RepresentationConfig)


class TestRunnerIntegration:
    def test_run_update_without_training(self, tmp_path: Path) -> None:
        config_path = tmp_path / "update.yaml"
        processed = tmp_path / "data" / "processed"
        report = tmp_path / "outputs" / "reports"
        processed.mkdir(parents=True)
        report.mkdir(parents=True)
        curve = make_synthetic_tess_curve(target_id="TIC 888001", n_per_sector=600, seed=3)
        save_light_curve(curve, processed / "tic_888001.npz")
        save_candidates([], report / "tce_candidates.json")
        config_path.write_text(
            yaml.dump(
                {
                    "experiment_name": "update_runner",
                    "seed": 0,
                    "paths": _paths_dict(tmp_path),
                    "logging": {"to_file": False},
                    "data": {
                        "source": {"name": "synthetic_tic", "params": {}},
                        "dataset": {"name": "synthetic_tess", "params": {}},
                        "train_fraction": 0.7,
                        "val_fraction": 0.15,
                        "stratify": True,
                    },
                    "preprocessing": {
                        "steps": [
                            {"name": "nan_removal", "params": {"strategy": "drop"}},
                            {"name": "normalize", "params": {"method": "median"}},
                        ]
                    },
                    "model": {
                        "architecture": {"name": "logistic_regression", "params": {}},
                        "features": [],
                    },
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
                    "grid": {
                        "name": "bls_auto",
                        "params": {
                            "min_period_days": 0.5,
                            "max_period_days": 8.0,
                            "oversample": 1.0,
                        },
                    },
                    "peaks": {"name": "prominence", "params": {"threshold_sigma": 5.0}},
                    "global_view": {
                        "name": "global",
                        "params": {"n_bins": 101, "max_empty_fraction": 0.7},
                    },
                    "local_view": {
                        "name": "local",
                        "params": {"n_bins": 41, "max_empty_fraction": 0.7},
                    },
                    "splitting": {
                        "name": "star",
                        "params": {"validation_fraction": 0.0, "test_fraction": 0.0},
                    },
                    "cache": {"enabled": False},
                    "update": {
                        "enabled": True,
                        "processed_dir": str(processed),
                        "resume_training": False,
                        "download": {"backend": "synthetic"},
                    },
                    "experiments": {"enabled": False},
                }
            ),
            encoding="utf-8",
        )
        from exodet.update.runner import run_update

        payload = run_update(config_path, processed_dir=str(processed))
        assert payload["update"]["n_success"] >= 1
        assert payload["training"] is None


class TestUpdateCli:
    def test_cli_processed_dir(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        processed = tmp_path / "processed"
        report = tmp_path / "reports"
        processed.mkdir()
        report.mkdir()
        curve = _injected_target(target_id="TIC 777001", seed=4)
        save_light_curve(curve, processed / "tic_777001.npz")
        save_candidates([], report / "tce_candidates.json")
        config_path = tmp_path / "update.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "experiment_name": "cli_update",
                    "seed": 0,
                    "paths": {
                        **_paths_dict(tmp_path),
                        "processed_dir": str(processed),
                        "report_dir": str(report),
                    },
                    "logging": {"to_file": False},
                    "data": {
                        "source": {"name": "synthetic_tic", "params": {}},
                        "dataset": {"name": "synthetic_tess", "params": {}},
                        "train_fraction": 0.7,
                        "val_fraction": 0.15,
                        "stratify": True,
                    },
                    "preprocessing": {
                        "steps": [
                            {"name": "nan_removal", "params": {"strategy": "drop"}},
                            {"name": "normalize", "params": {"method": "median"}},
                        ]
                    },
                    "model": {
                        "architecture": {"name": "logistic_regression", "params": {}},
                        "features": [],
                    },
                    "training": {
                        "trainer": {"name": "supervised", "params": {"backend": "sklearn"}},
                        "epochs": 1,
                        "batch_size": 8,
                        "learning_rate": 1e-3,
                        "early_stopping_patience": 0,
                    },
                    "evaluation": {"metrics": [], "decision_threshold": 0.5},
                    "grid": {"name": "bls_auto", "params": {"max_period_days": 8.0, "oversample": 1.0}},
                    "peaks": {"name": "prominence", "params": {"threshold_sigma": 5.0}},
                    "global_view": {"name": "global", "params": {"n_bins": 101, "max_empty_fraction": 0.7}},
                    "local_view": {"name": "local", "params": {"n_bins": 41, "max_empty_fraction": 0.7}},
                    "splitting": {
                        "name": "star",
                        "params": {"validation_fraction": 0.0, "test_fraction": 0.0},
                    },
                    "cache": {"enabled": False},
                    "update": {"enabled": True, "resume_training": False},
                    "experiments": {"enabled": False},
                }
            ),
            encoding="utf-8",
        )
        assert (
            main(
                [
                    "update",
                    "-c",
                    str(config_path),
                    "--processed",
                    str(processed),
                    "--force-reprocess",
                ]
            )
            == 0
        )
        assert "Update complete" in capsys.readouterr().out
