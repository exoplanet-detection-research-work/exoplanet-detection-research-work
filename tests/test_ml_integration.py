"""Integration tests for the supervised trainer and checkpointing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from exodet.config.schema import ExperimentConfig, ModelConfig, TrainingConfig
from exodet.ml.checkpoints import CheckpointManager
from exodet.ml.cross_validation import CrossValidationRunner
from exodet.ml.inference import InferenceEngine
from exodet.ml.models import XGBoostModel
from exodet.ml.trainer import SupervisedTrainer, build_trainer
from exodet.models.base import MODELS
from exodet.utils.seeding import seed_everything
from tests.ml_fixtures import LinearProbeModel, fast_training_config, make_representation_dataset

torch = pytest.importorskip("torch")


@pytest.fixture
def labeled_dataset():
    return make_representation_dataset(n_samples=32, n_stars=8, seed=42)


@pytest.fixture
def train_val_split(labeled_dataset):
    samples = labeled_dataset.samples
    train = type(labeled_dataset)(samples[:24], version="test")
    val = type(labeled_dataset)(samples[24:], version="test")
    return train, val


class TestSupervisedTrainer:
    def test_torch_training_produces_checkpoints(
        self, train_val_split, tmp_path: Path
    ) -> None:
        seed_everything(0)
        train, val = train_val_split
        config = TrainingConfig.from_dict(fast_training_config(epochs=4))
        trainer = SupervisedTrainer(config)
        model = MODELS.build("linear_probe")
        ckpt_dir = tmp_path / "ckpt"
        result = trainer.train(model, train, val, checkpoint_dir=ckpt_dir)
        assert (ckpt_dir / "last.pt").is_file()
        assert result.history["train_loss"]
        assert len(result.history["train_loss"]) >= 1

    def test_checkpoint_resume(
        self, train_val_split, tmp_path: Path
    ) -> None:
        seed_everything(1)
        train, val = train_val_split
        config = TrainingConfig.from_dict(fast_training_config(epochs=6))
        trainer = SupervisedTrainer(config)
        ckpt_dir = tmp_path / "ckpt"
        model1 = MODELS.build("linear_probe")
        trainer.train(model1, train, val, checkpoint_dir=ckpt_dir)
        model2 = MODELS.build("linear_probe")
        result = trainer.train(
            model2,
            train,
            val,
            checkpoint_dir=ckpt_dir,
            resume_from=ckpt_dir / "last.pt",
        )
        assert len(result.history["train_loss"]) < 6

    def test_deterministic_training(self, train_val_split) -> None:
        train, val = train_val_split
        config = TrainingConfig.from_dict(fast_training_config(epochs=2))

        def _run(seed: int) -> list[float]:
            seed_everything(seed)
            trainer = SupervisedTrainer(config)
            model = MODELS.build("linear_probe")
            result = trainer.train(model, train, val, checkpoint_dir=None)
            return result.history["train_loss"]

        assert _run(99) == _run(99)

    def test_xgboost_sklearn_backend(self, train_val_split) -> None:
        train, val = train_val_split
        raw = fast_training_config(
            trainer_params={"backend": "sklearn", "use_views": "features_only"}
        )
        config = TrainingConfig.from_dict(raw)
        trainer = SupervisedTrainer(config)
        model = XGBoostModel(n_estimators=10, max_depth=2)
        result = trainer.train(model, train, val, checkpoint_dir=None)
        probs = trainer.predict(model, val)
        assert len(probs) == len(val)
        assert np.all((probs >= 0) & (probs <= 1))


class TestInference:
    def test_batch_and_single(self, train_val_split, tmp_path: Path) -> None:
        train, val = train_val_split
        config = TrainingConfig.from_dict(fast_training_config(epochs=2))
        trainer = SupervisedTrainer(config)
        model = MODELS.build("linear_probe")
        ckpt = tmp_path / "ckpt"
        trainer.train(model, train, val, checkpoint_dir=ckpt)
        engine = InferenceEngine(model=model, trainer=trainer)
        batch_result = engine.predict_batch(val)
        single = engine.predict_single(val.samples[0])
        assert len(batch_result.probabilities) == len(val)
        assert len(single.probabilities) == 1


class TestCrossValidation:
    def test_group_kfold_splits(self, labeled_dataset) -> None:
        runner = CrossValidationRunner(
            {"enabled": True, "strategy": "group_kfold", "n_splits": 4}
        )
        splits = list(runner.splits(labeled_dataset))
        assert len(splits) == 4
        for split in splits:
            train_stars = {s.target_id for s in split.train.samples}
            val_stars = {s.target_id for s in split.validation.samples}
            assert train_stars.isdisjoint(val_stars)


class TestCheckpointManager:
    def test_top_k_pruning(self, tmp_path: Path) -> None:
        manager = CheckpointManager(directory=tmp_path, top_k=2, monitor="val_loss")
        for epoch, loss in enumerate([0.5, 0.4, 0.3], start=1):
            manager.save(
                epoch=epoch,
                model_state={"w": torch.tensor([float(loss)])},
                metrics={"val_loss": loss},
            )
        ranked = list(tmp_path.glob("epoch_*.pt"))
        assert len(ranked) <= 2
