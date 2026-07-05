"""Unit tests for ML registries and utilities."""

from __future__ import annotations

import numpy as np
import pytest

from exodet.config.schema import ComponentConfig, TrainingConfig
from exodet.exceptions import ConfigurationError, PipelineError
from exodet.ml.amp import AmpSettings, GradScalerManager
from exodet.ml.callbacks import (
    CALLBACKS,
    EarlyStoppingCallback,
    build_callbacks,
)
from exodet.ml.config import load_ml_settings
from exodet.ml.device import select_device
from exodet.ml.losses import LOSS_FUNCTIONS, build_loss
from exodet.ml.metrics import (
    CLASSIFICATION_METRICS,
    compute_all_metrics,
    expected_calibration_error,
)
from exodet.ml.optimizers import OPTIMIZERS, build_optimizer
from exodet.ml.schedulers import SCHEDULERS, build_scheduler
from tests.ml_fixtures import LinearProbeModel, fast_training_config


torch = pytest.importorskip("torch")


class TestMlSettings:
    def test_loads_extended_params(self) -> None:
        config = TrainingConfig.from_dict(fast_training_config())
        settings = load_ml_settings(config)
        assert settings.loss.name == "bce"
        assert settings.optimizer.name == "adamw"
        assert settings.amp == "none"
        assert settings.use_views == "both"

    def test_rejects_invalid_amp(self) -> None:
        raw = fast_training_config(trainer_params={"amp": "fp32"})
        with pytest.raises(ConfigurationError):
            load_ml_settings(TrainingConfig.from_dict(raw))


class TestLossRegistry:
    def test_builds_all_losses(self) -> None:
        for name in ("bce", "weighted_bce", "focal", "label_smooth_bce"):
            loss = build_loss(name)
            logits = torch.tensor([0.5, -0.2])
            targets = torch.tensor([1.0, 0.0])
            value = loss(logits, targets)
            assert float(value) >= 0.0


class TestOptimizerRegistry:
    def test_builds_optimizers(self) -> None:
        param = torch.nn.Parameter(torch.zeros(3))
        for name in ("adamw", "adam", "sgd", "rmsprop", "lion"):
            opt = build_optimizer(name, [param], lr=1e-3)
            assert opt is not None


class TestSchedulerRegistry:
    def test_builds_schedulers(self) -> None:
        param = torch.nn.Parameter(torch.zeros(3))
        opt = build_optimizer("adamw", [param], lr=1e-3)
        for name in ("cosine", "warm_restarts", "one_cycle", "plateau", "linear_warmup"):
            sched = build_scheduler(name, opt, epochs=5, steps_per_epoch=4)
            assert sched is not None


class TestAmp:
    def test_cpu_disables_amp(self) -> None:
        settings = AmpSettings.from_mode("fp16", "cpu")
        assert not settings.enabled

    def test_fp16_on_cuda_when_available(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        settings = AmpSettings.from_mode("fp16", "cuda")
        assert settings.enabled
        assert settings.use_scaler

    def test_grad_scaler_roundtrip(self) -> None:
        settings = AmpSettings(enabled=False, dtype=None, use_scaler=False)
        scaler = GradScalerManager(settings)
        assert scaler.state_dict() == {}
        scaler.load_state_dict({})


class TestDevice:
    def test_auto_selects_cpu(self) -> None:
        info = select_device("cpu")
        assert info.kind == "cpu"


class TestMetrics:
    def test_classification_metrics(self) -> None:
        labels = np.array([0, 1, 1, 0], dtype=np.int_)
        probs = np.array([0.1, 0.9, 0.8, 0.2], dtype=np.float64)
        specs = (
            ComponentConfig(name="accuracy"),
            ComponentConfig(name="roc_auc"),
            ComponentConfig(name="f1"),
        )
        scores, extra = compute_all_metrics(specs, labels, probs)
        assert scores["accuracy"] == 1.0
        assert scores["roc_auc"] == 1.0
        assert "confusion_matrix" not in extra

    def test_calibration_error(self) -> None:
        labels = np.array([0, 1, 0, 1], dtype=np.int_)
        probs = np.array([0.2, 0.8, 0.3, 0.7], dtype=np.float64)
        ece = expected_calibration_error(labels, probs, n_bins=2)
        assert 0.0 <= ece <= 1.0


class TestCallbacks:
    def test_early_stopping(self) -> None:
        from exodet.ml.trainer import TrainerState

        cb = EarlyStoppingCallback(patience=2)
        state = TrainerState()
        for value in (0.5, 0.4, 0.45, 0.46):
            state.epoch_metrics = {"val_loss": value}
            cb.on_epoch_end(state, 0)
        assert state.stop_training

    def test_build_callbacks_from_yaml(self) -> None:
        specs = (
            ComponentConfig(name="early_stopping", params={"patience": 2}),
            ComponentConfig(name="lr_monitor", params={}),
        )
        cbs = build_callbacks(specs, grad_clip_norm=0.5)
        assert len(cbs.callbacks) >= 2
