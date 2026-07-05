"""Tests for research training strategies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from exodet.config.schema import TrainingConfig
from exodet.ml.data import collate_ml_batch
from exodet.training.augmentation import (
    build_training_augmentation,
)
from exodet.training.benchmarking import benchmark_training_step
from exodet.training.calibration import (
    fit_temperature_scaling,
    reliability_bins,
)
from exodet.training.checkpoint_averaging import (
    ExponentialMovingAverage,
    StochasticWeightAveraging,
)
from exodet.training.config import load_research_config
from exodet.training.contrastive import NTXentLoss
from exodet.training.curriculum import (
    ClassImbalanceHandler,
    CurriculumScheduler,
    effective_number_weights,
)
from exodet.training.data import HardExampleTracker
from exodet.training.distillation import DistillationLoss, TeacherStudentSetup
from exodet.training.evaluation import ScientificValidator
from exodet.training.pretraining import run_masked_pretraining
from exodet.training.research_trainer import ResearchSupervisedTrainer
from exodet.utils.seeding import seed_everything
from tests.ml_fixtures import (
    fast_training_config,
    make_labeled_sample,
    make_representation_dataset,
)

torch = pytest.importorskip("torch")

import exodet.models.registry  # noqa: F401, E402
import exodet.training.research_trainer  # noqa: F401, E402
from exodet.ml.trainer import build_trainer
from exodet.models.base import MODELS


class TestResearchConfig:
    def test_load_from_training(self) -> None:
        raw = fast_training_config(
            trainer_params={
                "research": {
                    "enabled": True,
                    "curriculum": {"enabled": True},
                }
            }
        )
        raw["trainer"]["name"] = "research"
        cfg = load_research_config(TrainingConfig.from_dict(raw))
        assert cfg.enabled
        assert cfg.curriculum["enabled"]


class TestCurriculum:
    def test_snr_stages(self) -> None:
        dataset = make_representation_dataset(n_samples=20, seed=0)
        sched = CurriculumScheduler(enabled=True)
        idx = sched.allowed_indices(dataset, epoch=1, total_epochs=10)
        assert len(idx) > 0

    def test_effective_number_weights(self) -> None:
        w = effective_number_weights(np.array([100, 10, 5]))
        assert w[1] > w[0]


class TestImbalance:
    def test_weighted_sampler(self) -> None:
        dataset = make_representation_dataset(n_samples=16, seed=1)
        handler = ClassImbalanceHandler(enabled=True, strategy="weighted_sampler")
        sampler = handler.make_sampler(dataset, epoch=1)
        assert sampler is not None


class TestAugmentation:
    def test_training_augmentation(self) -> None:
        sample = make_labeled_sample(seed=0, n_global=32, n_local=16, n_features=8)
        pipeline = build_training_augmentation(
            {
                "enabled": True,
                "probability": 1.0,
                "steps": [{"name": "gaussian_noise", "params": {}}],
            }
        )
        rng = np.random.default_rng(0)
        augmented = pipeline.apply(sample, rng)
        assert not np.allclose(augmented.global_view, sample.global_view)


class TestHardMining:
    def test_tracker_boosts_hard_samples(self) -> None:
        dataset = make_representation_dataset(n_samples=8, seed=2)
        tracker = HardExampleTracker(enabled=True, boost=3.0)
        tracker.update(
            tuple(s.sample_id for s in dataset.samples),
            [0.1, 0.9, 0.2, 0.8, 0.1, 0.7, 0.2, 0.6],
            [0.9, 0.1, 0.8, 0.2, 0.9, 0.3, 0.85, 0.15],
        )
        weights = tracker.weights_for_dataset(dataset)
        assert weights.max() > 1.0


class TestDistillation:
    def test_distillation_loss_without_teacher(self) -> None:
        base = torch.nn.BCEWithLogitsLoss()
        setup = TeacherStudentSetup(enabled=False)
        loss_fn = DistillationLoss(base, setup, None)
        logits = torch.tensor([0.5, -0.2])
        labels = torch.tensor([1.0, 0.0])
        loss = loss_fn(logits, labels)
        assert float(loss) >= 0.0


class TestContrastive:
    def test_nt_xent(self) -> None:
        z = torch.randn(4, 8)
        loss = NTXentLoss(0.1)(z, z)
        assert float(loss) >= 0.0


class TestCalibration:
    def test_temperature_fit(self) -> None:
        logits = np.array([-1.0, 1.0, -0.5, 0.5, 2.0])
        labels = np.array([0, 1, 0, 1, 1], dtype=np.int_)
        scaler = fit_temperature_scaling(logits, labels, max_iter=20)
        assert scaler.temperature > 0
        probs = scaler.calibrate(logits)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_reliability_bins(self) -> None:
        labels = np.array([0, 1, 0, 1])
        probs = np.array([0.2, 0.8, 0.3, 0.7])
        stats = reliability_bins(labels, probs, n_bins=2)
        assert "ece" in stats


class TestCheckpointAveraging:
    def test_ema_update(self) -> None:
        ema = ExponentialMovingAverage(0.9)
        state = {"w": torch.tensor([1.0])}
        ema.update(state)
        ema.update({"w": torch.tensor([2.0])})
        assert ema.shadow is not None

    def test_swa_average(self) -> None:
        swa = StochasticWeightAveraging(start_epoch=1)
        swa.maybe_collect(1, {"w": torch.tensor([1.0])})
        swa.maybe_collect(2, {"w": torch.tensor([3.0])})
        avg = swa.average()
        assert avg is not None
        assert float(avg["w"]) == pytest.approx(2.0)


class TestEvaluation:
    def test_scientific_strata(self) -> None:
        dataset = make_representation_dataset(n_samples=12, seed=3)
        labels = np.array([s.label for s in dataset.samples])
        probs = np.linspace(0.1, 0.9, len(labels))
        validator = ScientificValidator()
        strata = validator.summarize(dataset, labels, probs)
        assert "short_period" in strata or "long_period" in strata


class TestResearchTrainer:
    def test_build_research_trainer(self) -> None:
        raw = fast_training_config(
            trainer_params={"research": {"enabled": True, "augmentation": {"enabled": False}}}
        )
        raw["trainer"]["name"] = "research"
        trainer = build_trainer(TrainingConfig.from_dict(raw))
        assert isinstance(trainer, ResearchSupervisedTrainer)

    def test_research_training_loop(self, tmp_path: Path) -> None:
        seed_everything(0)
        train = make_representation_dataset(n_samples=24, n_stars=6, n_global=32, n_local=16, n_features=8)
        val = make_representation_dataset(n_samples=8, n_stars=2, n_global=32, n_local=16, n_features=8)
        raw = fast_training_config(
            epochs=2,
            batch_size=4,
            trainer_params={
                "research": {
                    "enabled": True,
                    "curriculum": {"enabled": True},
                    "imbalance": {"enabled": True, "strategy": "weighted_sampler"},
                    "augmentation": {"enabled": True, "probability": 0.5, "steps": [{"name": "gaussian_noise", "params": {}}]},
                    "hard_mining": {"enabled": True},
                    "monitoring": {"enabled": True, "output_dir": str(tmp_path / "mon")},
                }
            },
        )
        raw["trainer"]["name"] = "research"
        params = {
            "global_bins": 32,
            "local_bins": 16,
            "embed_dim": 16,
            "hidden_dim": 32,
            "cnn_channels": [8, 16],
            "cnn_kernel_sizes": [5, 3],
            "transformer_depth": 1,
            "transformer_heads": 2,
            "physics_hidden_dims": [16],
            "num_classes": 5,
        }
        model = MODELS.build("fusion", **params)
        trainer = ResearchSupervisedTrainer(TrainingConfig.from_dict(raw))
        result = trainer.train(model, train, val, checkpoint_dir=tmp_path / "ckpt")
        assert result.history["train_loss"]


class TestDeterminism:
    def test_curriculum_deterministic(self) -> None:
        dataset = make_representation_dataset(n_samples=16, seed=5)
        sched = CurriculumScheduler(enabled=True)
        a = sched.allowed_indices(dataset, 2, 10)
        b = sched.allowed_indices(dataset, 2, 10)
        assert np.array_equal(a, b)


class TestPretraining:
    def test_masked_pretrain_exports_checkpoint(self, tmp_path: Path) -> None:
        seed_everything(0)
        dataset = make_representation_dataset(n_samples=12, n_global=32, n_local=16, n_features=8)
        params = {
            "global_bins": 32,
            "local_bins": 16,
            "embed_dim": 16,
            "hidden_dim": 32,
            "cnn_channels": [8, 16],
            "cnn_kernel_sizes": [5, 3],
            "transformer_depth": 1,
            "transformer_heads": 2,
            "physics_hidden_dims": [16],
            "num_classes": 5,
        }
        model = MODELS.build("fusion", **params)
        ckpt = run_masked_pretraining(
            model,
            dataset,
            {"epochs": 1, "batch_size": 4, "mask_fraction": 0.2},
            tmp_path / "pretrain",
        )
        assert ckpt.is_file()
        loaded = type(model).load(ckpt)
        assert loaded is not None


class TestBenchmarking:
    def test_benchmark_training_step(self) -> None:
        params = {
            "global_bins": 32,
            "local_bins": 16,
            "embed_dim": 16,
            "hidden_dim": 32,
            "cnn_channels": [8, 16],
            "cnn_kernel_sizes": [5, 3],
            "transformer_depth": 1,
            "transformer_heads": 2,
            "physics_hidden_dims": [16],
            "num_classes": 5,
        }
        model = MODELS.build("fusion", **params)
        items = [make_labeled_sample(seed=i, n_global=32, n_local=16, n_features=8) for i in range(4)]
        batch_dicts = [
            {
                "global_view": s.global_view,
                "local_view": s.local_view,
                "features": s.features,
                "labels": s.label,
                "weights": s.weight,
                "sample_id": s.sample_id,
                "target_id": s.target_id,
            }
            for s in items
        ]
        batch = collate_ml_batch(batch_dicts, use_views="both")
        model._ensure_module(32 + 16 + 8, torch.device("cpu"))
        result = benchmark_training_step(model, batch, "cpu", n_steps=2)
        assert result.steps_per_second > 0


class TestCheckpointCompatibility:
    def test_research_trainer_checkpoint_roundtrip(self, tmp_path: Path) -> None:
        seed_everything(1)
        train = make_representation_dataset(n_samples=12, n_global=32, n_local=16, n_features=8)
        val = make_representation_dataset(n_samples=4, n_global=32, n_local=16, n_features=8)
        raw = fast_training_config(
            epochs=1,
            batch_size=4,
            trainer_params={"research": {"enabled": True, "augmentation": {"enabled": False}}},
        )
        raw["trainer"]["name"] = "research"
        params = {
            "global_bins": 32,
            "local_bins": 16,
            "embed_dim": 16,
            "hidden_dim": 32,
            "cnn_channels": [8, 16],
            "cnn_kernel_sizes": [5, 3],
            "transformer_depth": 1,
            "transformer_heads": 2,
            "physics_hidden_dims": [16],
            "num_classes": 5,
        }
        model = MODELS.build("fusion", **params)
        trainer = ResearchSupervisedTrainer(TrainingConfig.from_dict(raw))
        ckpt_dir = tmp_path / "ckpt"
        trainer.train(model, train, val, checkpoint_dir=ckpt_dir)
        best = sorted(ckpt_dir.glob("*.pt"))
        assert best
        restored = type(model).load(best[-1])
        assert restored._fitted or restored._module is not None


class TestDistillationWithTeacher:
    def test_distillation_with_mock_teacher(self) -> None:
        base = torch.nn.BCEWithLogitsLoss()
        setup = TeacherStudentSetup(enabled=True, temperature=2.0, alpha=0.5)

        class _Teacher:
            def forward_batch(self, batch: object) -> torch.Tensor:
                return torch.zeros(2)

        loss_fn = DistillationLoss(base, setup, _Teacher())
        logits = torch.tensor([0.3, -0.1])
        labels = torch.tensor([1.0, 0.0])
        batch = collate_ml_batch(
            [
                {
                    "global_view": np.zeros(32, dtype=np.float32),
                    "local_view": np.zeros(16, dtype=np.float32),
                    "features": np.zeros(8, dtype=np.float32),
                    "labels": 1,
                    "weights": 1.0,
                    "sample_id": "a",
                    "target_id": "t1",
                },
                {
                    "global_view": np.zeros(32, dtype=np.float32),
                    "local_view": np.zeros(16, dtype=np.float32),
                    "features": np.zeros(8, dtype=np.float32),
                    "labels": 0,
                    "weights": 1.0,
                    "sample_id": "b",
                    "target_id": "t2",
                },
            ],
            use_views="both",
        )
        loss = loss_fn(logits, labels, batch)
        assert float(loss) >= 0.0
