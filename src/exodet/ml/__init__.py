"""Deep-learning training infrastructure.

Provides the reusable ecosystem that future CNN, Transformer, and
fusion models plug into. This package does **not** ship concrete
architectures — only registries, the supervised trainer, checkpointing,
callbacks, mixed precision, metrics, cross-validation, and inference.

Registries (selectable from YAML via ``training.trainer.params``):

==================  ==================  ============================
Component           Registry            Implementations
==================  ==================  ============================
Loss                ``LOSS_FUNCTIONS``  bce, weighted_bce, focal, label_smooth_bce
Optimizer           ``OPTIMIZERS``      adamw, adam, sgd, rmsprop, lion
Scheduler           ``SCHEDULERS``      cosine, warm_restarts, one_cycle, plateau, linear_warmup
Callback            ``CALLBACKS``       early_stopping, checkpoint, lr_monitor, grad_clip, predict_export
==================  ==================  ============================

The :class:`~exodet.ml.trainer.SupervisedTrainer` is registered as
``supervised`` in :data:`~exodet.training.base.TRAINERS`.
"""

from __future__ import annotations

from exodet.ml.amp import AmpSettings, GradScalerManager
from exodet.ml.callbacks import (
    CALLBACKS,
    Callback,
    CallbackList,
    CheckpointCallback,
    EarlyStoppingCallback,
    GradientClippingCallback,
    LearningRateMonitorCallback,
    PredictionExportCallback,
)
from exodet.ml.checkpoints import CheckpointManager
from exodet.ml.config import MlSettings, load_ml_settings
from exodet.ml.cross_validation import CrossValidationRunner, CvSplit
from exodet.ml.data import MlBatch, RepresentationDataModule
from exodet.ml.device import DeviceInfo, select_device
from exodet.ml.inference import InferenceEngine, InferenceResult
from exodet.ml.losses import LOSS_FUNCTIONS, build_loss
from exodet.ml.metrics import CLASSIFICATION_METRICS, compute_all_metrics
from exodet.ml.models import MODEL_BACKENDS, BaseTorchModel, XGBoostModel
from exodet.ml.optimizers import OPTIMIZERS, build_optimizer
from exodet.ml.runner import run_evaluation, run_predict, run_training
from exodet.ml.schedulers import SCHEDULERS, build_scheduler
from exodet.ml.tracking import CsvLogger, ExperimentTracker, TensorBoardLogger
from exodet.ml.trainer import SupervisedTrainer

__all__ = [
    "LOSS_FUNCTIONS",
    "OPTIMIZERS",
    "SCHEDULERS",
    "CALLBACKS",
    "MODEL_BACKENDS",
    "CLASSIFICATION_METRICS",
    "AmpSettings",
    "GradScalerManager",
    "BaseTorchModel",
    "XGBoostModel",
    "Callback",
    "CallbackList",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "GradientClippingCallback",
    "LearningRateMonitorCallback",
    "PredictionExportCallback",
    "CheckpointManager",
    "MlSettings",
    "load_ml_settings",
    "CrossValidationRunner",
    "CvSplit",
    "MlBatch",
    "RepresentationDataModule",
    "DeviceInfo",
    "select_device",
    "InferenceEngine",
    "InferenceResult",
    "build_loss",
    "compute_all_metrics",
    "build_optimizer",
    "build_scheduler",
    "run_training",
    "run_evaluation",
    "run_predict",
    "CsvLogger",
    "ExperimentTracker",
    "TensorBoardLogger",
    "SupervisedTrainer",
]
