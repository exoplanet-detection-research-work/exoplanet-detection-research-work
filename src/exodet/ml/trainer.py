"""Supervised training engine (Module 2)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from exodet.config.schema import TrainingConfig
from exodet.exceptions import PipelineError
from exodet.ml.amp import AmpSettings, GradScalerManager
from exodet.ml.callbacks import CallbackList, build_callbacks
from exodet.ml.checkpoints import CheckpointManager
from exodet.ml.config import MlSettings, load_ml_settings
from exodet.ml.data import MlBatch, RepresentationDataModule
from exodet.ml.device import DeviceInfo, select_device
from exodet.ml.losses import build_loss
from exodet.ml.metrics import compute_all_metrics
from exodet.ml.models import BaseTorchModel, XGBoostModel
from exodet.ml.optimizers import build_optimizer
from exodet.ml.schedulers import build_scheduler
from exodet.ml.tracking import build_tracker
from exodet.models.base import MODELS, BaseModel
from exodet.representation.containers import RepresentationDataset
from exodet.training.base import TRAINERS, BaseTrainer, TrainingResult
from exodet.utils.io import write_json

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["TrainerState", "SupervisedTrainer"]

logger = logging.getLogger(__name__)


def _require_torch():
    import torch

    return torch


@dataclass
class TrainerState:
    """Mutable training state shared with callbacks.

    Attributes:
        stop_training: Set ``True`` by early stopping to halt epochs.
        epoch_metrics: Metrics from the current epoch.
        best_checkpoint: Path to the best checkpoint so far.
        config_snapshot: Frozen config for checkpointing.
        val_probabilities: Last-epoch validation probabilities.
        val_labels: Validation labels.
        val_sample_ids: Validation sample ids.
        val_target_ids: Validation target ids.
    """

    stop_training: bool = False
    epoch_metrics: dict[str, float] = field(default_factory=dict)
    best_checkpoint: Path | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    val_probabilities: list[float] = field(default_factory=list)
    val_labels: list[int] = field(default_factory=list)
    val_sample_ids: list[str] = field(default_factory=list)
    val_target_ids: list[str] = field(default_factory=list)

    optimizer: Any = None
    scheduler: Any = None
    scaler: GradScalerManager | None = None
    model_module: Any = None

    def model_state_dict(self) -> dict[str, Any]:
        if self.model_module is None:
            return {}
        return self.model_module.state_dict()

    def optimizer_state_dict(self) -> dict[str, Any]:
        if self.optimizer is None:
            return {}
        return self.optimizer.state_dict()

    def scheduler_state_dict(self) -> dict[str, Any]:
        if self.scheduler is None:
            return {}
        return self.scheduler.state_dict()

    def scaler_state_dict(self) -> dict[str, Any]:
        if self.scaler is None:
            return {}
        return self.scaler.state_dict()


@TRAINERS.register("supervised")
class SupervisedTrainer(BaseTrainer):
    """Config-driven trainer for torch and sklearn backends.

    Supports training, validation, testing, prediction, checkpoint
    resume, automatic device selection, and mixed precision.
    """

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__(config)
        self.ml_settings = load_ml_settings(config)

    def train(
        self,
        model: BaseModel,
        train_data: RepresentationDataset,
        val_data: RepresentationDataset | None = None,
        checkpoint_dir: Path | None = None,
        resume_from: Path | None = None,
    ) -> TrainingResult:
        """Runs the full training procedure.

        Args:
            model: Model to train.
            train_data: Training representation dataset.
            val_data: Optional validation dataset.
            checkpoint_dir: Directory for checkpoints.
            resume_from: Optional checkpoint to resume from.

        Returns:
            Training result with history and best checkpoint path.
        """
        if self.ml_settings.backend == "sklearn":
            return self._train_sklearn(model, train_data, val_data, checkpoint_dir)
        if isinstance(model, BaseTorchModel):
            return self._train_torch(
                model, train_data, val_data, checkpoint_dir, resume_from
            )
        raise PipelineError(
            f"Model {type(model).__name__} is incompatible with backend "
            f"'{self.ml_settings.backend}'."
        )

    def _train_sklearn(
        self,
        model: BaseModel,
        train_data: RepresentationDataset,
        val_data: RepresentationDataset | None,
        checkpoint_dir: Path | None,
    ) -> TrainingResult:
        arrays = train_data.to_numpy()
        mask = arrays["labels"] >= 0
        features = self._flatten_numpy(arrays, mask)
        labels = arrays["labels"][mask].astype(np.int_)
        val_pair = None
        if val_data is not None and len(val_data) > 0:
            val_arrays = val_data.to_numpy()
            val_mask = val_arrays["labels"] >= 0
            val_pair = (
                self._flatten_numpy(val_arrays, val_mask),
                val_arrays["labels"][val_mask].astype(np.int_),
            )
        if isinstance(model, XGBoostModel) or hasattr(model, "fit"):
            model.fit(features, labels, validation=val_pair)
        else:
            raise PipelineError(f"Sklearn backend cannot train {type(model).__name__}.")
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save(checkpoint_dir / "model.json")
        return TrainingResult(model=model, best_checkpoint=checkpoint_dir)

    def _flatten_numpy(self, arrays: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
        parts = []
        use_views = self.ml_settings.use_views
        if use_views in ("global", "both"):
            parts.append(arrays["global_view"][mask])
        if use_views in ("local", "both"):
            parts.append(arrays["local_view"][mask])
        if use_views in ("global", "local", "both", "features_only"):
            parts.append(arrays["features"][mask])
        return np.concatenate(parts, axis=1).astype(np.float64)

    def _train_torch(
        self,
        model: BaseTorchModel,
        train_data: RepresentationDataset,
        val_data: RepresentationDataset | None,
        checkpoint_dir: Path | None,
        resume_from: Path | None,
    ) -> TrainingResult:
        torch = _require_torch()
        settings = self.ml_settings
        device_pref = str(settings.checkpoint.get("device", "auto"))
        device_info = select_device(device_pref)
        device = device_info.device

        data_module = RepresentationDataModule(
            train=train_data,
            validation=val_data,
            batch_size=self.config.batch_size,
            num_workers=settings.num_workers,
            pin_memory=settings.pin_memory and device_info.kind == "cuda",
            use_views=settings.use_views,
        )
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        steps_per_epoch = max(1, len(train_loader))

        sample_batch = next(iter(train_loader))
        input_dim = self._input_dim_from_batch(sample_batch)
        model._ensure_module(input_dim, device)

        loss_fn = build_loss(settings.loss.name, **settings.loss.params)
        loss_fn.to(device)
        optimizer = build_optimizer(
            settings.optimizer.name,
            model.module.parameters(),
            lr=self.config.learning_rate,
            **settings.optimizer.params,
        )
        scheduler = None
        if settings.scheduler is not None:
            scheduler = build_scheduler(
                settings.scheduler.name,
                optimizer,
                epochs=self.config.epochs,
                steps_per_epoch=steps_per_epoch,
                **settings.scheduler.params,
            )

        amp_settings = AmpSettings.from_mode(settings.amp, device_info.kind)
        scaler = GradScalerManager(amp_settings)

        ckpt_cfg = settings.checkpoint
        manager = None
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            manager = CheckpointManager(
                directory=checkpoint_dir,
                monitor=str(ckpt_cfg.get("monitor", "val_loss")),
                mode=str(ckpt_cfg.get("mode", "min")),
                save_last=bool(ckpt_cfg.get("save_last", True)),
                save_best=bool(ckpt_cfg.get("save_best", True)),
                top_k=int(ckpt_cfg.get("top_k", 3)),
            )

        callbacks = build_callbacks(
            settings.callbacks,
            checkpoint_manager=manager,
            grad_clip_norm=settings.grad_clip_norm,
        )
        if self.config.early_stopping_patience > 0 and not any(
            c.__class__.__name__ == "EarlyStoppingCallback"
            for c in callbacks.callbacks
        ):
            from exodet.ml.callbacks import EarlyStoppingCallback

            callbacks.append(
                EarlyStoppingCallback(
                    monitor=str(ckpt_cfg.get("monitor", "val_loss")),
                    mode=str(ckpt_cfg.get("mode", "min")),
                    patience=self.config.early_stopping_patience,
                )
            )

        tracker = build_tracker(
            settings.tracking,
            output_dir=checkpoint_dir or Path("outputs/training"),
            experiment_name="training",
            hyperparameters=self.describe(),
        )

        state = TrainerState(
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            model_module=model.module,
            config_snapshot={**self.describe(), "input_dim": input_dim},
        )
        start_epoch = 1
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        if resume_from is not None and resume_from.is_file():
            payload = torch.load(resume_from, map_location=device, weights_only=False)
            model.load_state_dict(payload["model_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            if scheduler is not None and payload.get("scheduler_state"):
                scheduler.load_state_dict(payload["scheduler_state"])
            scaler.load_state_dict(payload.get("scaler_state", {}))
            start_epoch = int(payload.get("epoch", 0)) + 1
            logger.info("Resumed training from epoch %d", start_epoch)

        callbacks.on_train_begin(state)
        for epoch in range(start_epoch, self.config.epochs + 1):
            if state.stop_training:
                break
            callbacks.on_epoch_begin(state, epoch)
            train_loss = self._train_epoch(
                model, train_loader, loss_fn, optimizer, scheduler,
                amp_settings, scaler, device, device_info.kind, state, epoch,
            )
            state.epoch_metrics = {"train_loss": train_loss}
            if val_loader is not None:
                val_loss, val_probs, val_labels, sids, tids = self._validate_epoch(
                    model, val_loader, loss_fn, device, amp_settings, device_info.kind
                )
                state.epoch_metrics["val_loss"] = val_loss
                state.val_probabilities = val_probs
                state.val_labels = val_labels
                state.val_sample_ids = sids
                state.val_target_ids = tids
                metric_specs = ()
                scores, _ = compute_all_metrics(
                    metric_specs,
                    np.array(val_labels, dtype=np.int_),
                    np.array(val_probs, dtype=np.float64),
                )
                state.epoch_metrics.update(scores)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(state.epoch_metrics.get("val_loss", float("nan")))
            tracker.log_metrics(state.epoch_metrics, step=epoch)
            callbacks.on_epoch_end(state, epoch)

        callbacks.on_train_end(state)
        tracker.close()
        model._fitted = True
        return TrainingResult(
            model=model,
            best_checkpoint=state.best_checkpoint or (manager.best_checkpoint if manager else None),
            history=history,
        )

    def _input_dim_from_batch(self, batch: MlBatch) -> int:
        parts = []
        if batch.global_view is not None:
            parts.append(batch.global_view.shape[1])
        if batch.local_view is not None:
            parts.append(batch.local_view.shape[1])
        if batch.features is not None:
            parts.append(batch.features.shape[1])
        return int(sum(parts))

    def _move_batch(self, batch: MlBatch, device: "torch.device") -> MlBatch:
        def _mv(t: "torch.Tensor | None") -> "torch.Tensor | None":
            return None if t is None else t.to(device)

        return MlBatch(
            global_view=_mv(batch.global_view),
            local_view=_mv(batch.local_view),
            features=_mv(batch.features),
            labels=batch.labels.to(device),
            weights=batch.weights.to(device),
            sample_ids=batch.sample_ids,
            target_ids=batch.target_ids,
        )

    def _train_epoch(
        self,
        model: BaseTorchModel,
        loader: Any,
        loss_fn: Any,
        optimizer: Any,
        scheduler: Any,
        amp_settings: AmpSettings,
        scaler: GradScalerManager,
        device: "torch.device",
        device_kind: str,
        state: TrainerState,
        epoch: int,
    ) -> float:
        del epoch
        torch = _require_torch()
        model.module.train()
        total_loss = 0.0
        n_batches = 0
        one_cycle = (
            self.ml_settings.scheduler is not None
            and self.ml_settings.scheduler.name.lower() == "one_cycle"
        )
        for batch_idx, batch in enumerate(loader):
            state.model_module = model.module
            batch = self._move_batch(batch, device)
            mask = batch.labels >= 0
            if not mask.any():
                continue
            labels = batch.labels[mask].float()
            weights = batch.weights[mask]
            sub_batch = self._subset_batch(batch, mask)

            optimizer.zero_grad(set_to_none=True)
            with amp_settings.autocast(device_kind):
                logits = model.forward_batch(sub_batch)
                loss = loss_fn(logits, labels)
                loss = (loss * weights).mean()

            scaled = scaler.scale(loss)
            scaled.backward()
            scaler.step(optimizer)
            if one_cycle and scheduler is not None:
                scheduler.step()

            total_loss += float(loss.detach().cpu())
            n_batches += 1

        if scheduler is not None and not one_cycle:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                pass
            else:
                scheduler.step()

        return total_loss / max(1, n_batches)

    def _subset_batch(self, batch: MlBatch, mask: "torch.Tensor") -> MlBatch:
        def _sel(t: "torch.Tensor | None") -> "torch.Tensor | None":
            return None if t is None else t[mask]

        ids = [sid for sid, m in zip(batch.sample_ids, mask.cpu().tolist(), strict=True) if m]
        tids = [tid for tid, m in zip(batch.target_ids, mask.cpu().tolist(), strict=True) if m]
        return MlBatch(
            global_view=_sel(batch.global_view),
            local_view=_sel(batch.local_view),
            features=_sel(batch.features),
            labels=batch.labels[mask],
            weights=batch.weights[mask],
            sample_ids=tuple(ids),
            target_ids=tuple(tids),
        )

    def _validate_epoch(
        self,
        model: BaseTorchModel,
        loader: Any,
        loss_fn: Any,
        device: "torch.device",
        amp_settings: AmpSettings,
        device_kind: str,
    ) -> tuple[float, list[float], list[int], list[str], list[str]]:
        torch = _require_torch()
        model.module.eval()
        total_loss = 0.0
        n_batches = 0
        probs: list[float] = []
        labels: list[int] = []
        sample_ids: list[str] = []
        target_ids: list[str] = []

        with torch.no_grad():
            for batch in loader:
                batch = self._move_batch(batch, device)
                mask = batch.labels >= 0
                if not mask.any():
                    continue
                sub = self._subset_batch(batch, mask)
                with amp_settings.autocast(device_kind):
                    logits = model.forward_batch(sub)
                    loss = loss_fn(logits, sub.labels.float())
                total_loss += float(loss.detach().cpu())
                n_batches += 1
                batch_probs = torch.sigmoid(logits).cpu().tolist()
                probs.extend(batch_probs)
                labels.extend(sub.labels.cpu().tolist())
                sample_ids.extend(sub.sample_ids)
                target_ids.extend(sub.target_ids)

        return (
            total_loss / max(1, n_batches),
            probs,
            labels,
            sample_ids,
            target_ids,
        )

    def predict(
        self,
        model: BaseModel,
        dataset: RepresentationDataset,
        batch_size: int | None = None,
    ) -> npt.NDArray[np.float64]:
        """Runs inference on a representation dataset.

        Args:
            model: Trained model.
            dataset: Samples to score.
            batch_size: Override batch size.

        Returns:
            Positive-class probabilities.
        """
        arrays = dataset.to_numpy()
        features = self._flatten_numpy(arrays, np.ones(len(arrays["labels"]), dtype=bool))
        if isinstance(model, BaseTorchModel):
            return model.predict_proba(features)
        return model.predict_proba(features)

    def evaluate(
        self,
        model: BaseModel,
        dataset: RepresentationDataset,
        metric_names: tuple[Any, ...] = (),
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """Evaluates a model on a dataset split.

        Args:
            model: Trained model.
            dataset: Evaluation samples.
            metric_names: Metric component configs.
            threshold: Decision threshold.

        Returns:
            Metric name to value mapping.
        """
        arrays = dataset.to_numpy()
        mask = arrays["labels"] >= 0
        probs = self.predict(model, dataset)[mask]
        labels = arrays["labels"][mask].astype(np.int_)
        scores, _ = compute_all_metrics(metric_names, labels, probs, threshold=threshold)
        return scores

    def describe(self) -> dict[str, Any]:
        """Summarizes trainer setup for provenance."""
        base = super().describe()
        base.update(
            {
                "backend": self.ml_settings.backend,
                "loss": self.ml_settings.loss.name,
                "optimizer": self.ml_settings.optimizer.name,
                "scheduler": (
                    self.ml_settings.scheduler.name
                    if self.ml_settings.scheduler
                    else None
                ),
                "amp": self.ml_settings.amp,
                "use_views": self.ml_settings.use_views,
            }
        )
        return base


def build_trainer(config: TrainingConfig) -> SupervisedTrainer:
    """Builds the supervised trainer from experiment config.

    Args:
        config: Training configuration.

    Returns:
        A configured :class:`SupervisedTrainer` or research extension.
    """
    from exodet.training.config import load_research_config

    research = load_research_config(config)
    name = config.trainer.name.lower()
    if name == "research" or research.enabled:
        from exodet.training.research_trainer import ResearchSupervisedTrainer

        return ResearchSupervisedTrainer(config)
    if name not in ("supervised", "torch_trainer"):
        return TRAINERS.build(name, config=config)
    return SupervisedTrainer(config)
