"""Research-grade supervised trainer extending the existing training loop."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from exodet.ml.amp import AmpSettings, GradScalerManager
from exodet.ml.callbacks import build_callbacks
from exodet.ml.checkpoints import CheckpointManager
from exodet.ml.device import select_device
from exodet.ml.losses import build_loss
from exodet.ml.metrics import compute_all_metrics
from exodet.ml.models import BaseTorchModel
from exodet.ml.optimizers import build_optimizer
from exodet.ml.schedulers import build_scheduler
from exodet.ml.tracking import build_tracker
from exodet.ml.trainer import SupervisedTrainer, TrainerState
from exodet.representation.containers import RepresentationDataset
from exodet.training.base import TRAINERS, TrainingResult
from exodet.training.checkpoint_averaging import EMACallback, SWACallback  # noqa: F401
from exodet.training.config import load_research_config
from exodet.training.data import HardExampleTracker, ResearchDataModule
from exodet.training.distillation import DistillationLoss, TeacherStudentSetup
from exodet.training.monitoring import build_research_callbacks

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["ResearchSupervisedTrainer"]

logger = logging.getLogger(__name__)


@TRAINERS.register("research")
class ResearchSupervisedTrainer(SupervisedTrainer):
    """Extends :class:`~exodet.ml.trainer.SupervisedTrainer` with research strategies.

    Public API matches ``SupervisedTrainer.train()``; research features are
    activated via ``training.trainer.params.research`` or trainer name
    ``research``.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.research = load_research_config(config)

    def train(
        self,
        model: Any,
        train_data: RepresentationDataset,
        val_data: RepresentationDataset | None = None,
        checkpoint_dir: Path | None = None,
        resume_from: Path | None = None,
    ) -> TrainingResult:
        if not isinstance(model, BaseTorchModel):
            return super().train(model, train_data, val_data, checkpoint_dir, resume_from)
        if not self.research.enabled and self.config.trainer.name.lower() != "research":
            return super().train(model, train_data, val_data, checkpoint_dir, resume_from)
        return self._train_torch_research(
            model, train_data, val_data, checkpoint_dir, resume_from
        )

    def _train_torch_research(
        self,
        model: BaseTorchModel,
        train_data: RepresentationDataset,
        val_data: RepresentationDataset | None,
        checkpoint_dir: Path | None,
        resume_from: Path | None,
    ) -> TrainingResult:
        import torch

        settings = self.ml_settings
        device_info = select_device(str(settings.checkpoint.get("device", "auto")))
        device = device_info.device

        hard_tracker = HardExampleTracker(
            enabled=bool(self.research.hard_mining.get("enabled", False)),
            boost=float(self.research.hard_mining.get("boost", 2.0)),
        )

        data_module = ResearchDataModule(
            train=train_data,
            research=self.research,
            batch_size=self.config.batch_size,
            num_workers=settings.num_workers,
            pin_memory=settings.pin_memory and device_info.kind == "cuda",
            use_views=settings.use_views,
            epoch=1,
            total_epochs=self.config.epochs,
            hard_tracker=hard_tracker,
        )
        val_loader = None
        if val_data is not None and len(val_data) > 0:
            from exodet.ml.data import RepresentationDataModule

            val_loader = RepresentationDataModule(
                train=val_data,
                batch_size=self.config.batch_size,
                num_workers=settings.num_workers,
                use_views=settings.use_views,
                shuffle_train=False,
            ).train_dataloader()

        train_loader = data_module.train_dataloader()
        sample_batch = next(iter(train_loader))
        input_dim = self._input_dim_from_batch(sample_batch)
        model._ensure_module(input_dim, device)

        base_loss = build_loss(settings.loss.name, **settings.loss.params)
        base_loss.to(device)
        distill_setup = TeacherStudentSetup(
            enabled=bool(self.research.distillation.get("enabled", False)),
            teacher_checkpoint=self.research.distillation.get("teacher_checkpoint"),
            temperature=float(self.research.distillation.get("temperature", 4.0)),
            alpha=float(self.research.distillation.get("alpha", 0.5)),
        )
        teacher = None
        if distill_setup.enabled:
            teacher = distill_setup.load_teacher(model)
            teacher._ensure_module(input_dim, device)
        loss_fn = DistillationLoss(base_loss, distill_setup, teacher)

        optimizer = build_optimizer(
            settings.optimizer.name,
            model.module.parameters(),
            lr=self.config.learning_rate,
            **settings.optimizer.params,
        )
        scheduler = None
        steps_per_epoch = max(1, len(train_loader))
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
        for cb in build_research_callbacks(
            asdict(self.research),
            hard_tracker=hard_tracker,
            output_dir=checkpoint_dir,
        ):
            callbacks.append(cb)

        avg_cfg = self.research.checkpoint_averaging
        if avg_cfg.get("ema", {}).get("enabled", False):
            callbacks.append(
                EMACallback(
                    decay=float(avg_cfg["ema"].get("decay", 0.999)),
                    export_path=str(checkpoint_dir / "ema.pt") if checkpoint_dir else None,
                )
            )
        if avg_cfg.get("swa", {}).get("enabled", False):
            callbacks.append(
                SWACallback(
                    start_epoch=int(avg_cfg["swa"].get("start_epoch", 5)),
                    export_path=str(checkpoint_dir / "swa.pt") if checkpoint_dir else None,
                )
            )

        tracker = build_tracker(
            settings.tracking,
            output_dir=checkpoint_dir or Path("outputs/training"),
            experiment_name="research_training",
            hyperparameters={**self.describe(), "research": True},
        )

        state = TrainerState(
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            model_module=model.module,
            config_snapshot={**self.describe(), "input_dim": input_dim, "research": True},
        )
        start_epoch = 1
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        if resume_from is not None and resume_from.is_file():
            payload = torch.load(resume_from, map_location=device, weights_only=False)
            model.load_state_dict(payload["model_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            if scheduler is not None and payload.get("scheduler_state"):
                scheduler.load_state_dict(payload["scheduler_state"])
            scaler.load_state_dict(payload.get("scaler_state", {}))
            start_epoch = int(payload.get("epoch", 0)) + 1

        callbacks.on_train_begin(state)
        for epoch in range(start_epoch, self.config.epochs + 1):
            if state.stop_training:
                break
            data_module.epoch = epoch
            train_loader = data_module.train_dataloader()
            callbacks.on_epoch_begin(state, epoch)
            train_loss = self._research_train_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                scheduler,
                amp_settings,
                scaler,
                device,
                device_info.kind,
                state,
                teacher,
                hard_tracker,
            )
            state.epoch_metrics = {"train_loss": train_loss}
            if val_loader is not None:
                val_loss, val_probs, val_labels, sids, tids = self._validate_epoch(
                    model, val_loader, base_loss, device, amp_settings, device_info.kind
                )
                state.epoch_metrics["val_loss"] = val_loss
                state.val_probabilities = val_probs
                state.val_labels = val_labels
                state.val_sample_ids = sids
                state.val_target_ids = tids
                scores, _ = compute_all_metrics(
                    (),
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

    def _research_train_epoch(
        self,
        model: BaseTorchModel,
        loader: Any,
        loss_fn: Any,
        optimizer: Any,
        scheduler: Any,
        amp_settings: AmpSettings,
        scaler: GradScalerManager,
        device: torch.device,
        device_kind: str,
        state: TrainerState,
        teacher: Any,
        hard_tracker: HardExampleTracker,
    ) -> float:
        import torch

        model.module.train()
        total_loss = 0.0
        n_batches = 0
        batch_losses: list[float] = []
        batch_ids: list[str] = []
        batch_confs: list[float] = []

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
                if isinstance(loss_fn, DistillationLoss):
                    loss = loss_fn(logits, labels, sub_batch)
                else:
                    loss = loss_fn(logits, labels)
                loss = (loss * weights).mean()

            scaled = scaler.scale(loss)
            scaled.backward()
            scaler.step(optimizer)
            if scheduler is not None and isinstance(
                scheduler, torch.optim.lr_scheduler.OneCycleLR
            ):
                scheduler.step()

            loss_val = float(loss.detach().cpu())
            total_loss += loss_val
            n_batches += 1
            batch_losses.extend([loss_val] * len(sub_batch.sample_ids))
            batch_ids.extend(sub_batch.sample_ids)
            with torch.no_grad():
                confs = torch.sigmoid(logits).cpu().tolist()
            batch_confs.extend(confs)
            state.epoch_metrics["train_batch_loss"] = loss_val

        if scheduler is not None and not isinstance(
            scheduler, torch.optim.lr_scheduler.OneCycleLR
        ):
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

        if hard_tracker.enabled and batch_ids:
            hard_tracker.update(tuple(batch_ids), batch_losses, batch_confs)

        return total_loss / max(1, n_batches)
