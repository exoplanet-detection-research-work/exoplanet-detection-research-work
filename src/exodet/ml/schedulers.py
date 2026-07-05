"""Learning-rate scheduler registry (Module 5)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from exodet.exceptions import PipelineError
from exodet.registry import Registry

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["SCHEDULERS", "build_scheduler"]

SCHEDULERS: Registry[object] = Registry("scheduler")


def _require_torch():
    import torch

    return torch


@SCHEDULERS.register("cosine")
class _CosineBuilder:
    def __call__(
        self,
        optimizer: "torch.optim.Optimizer",
        epochs: int,
        eta_min: float = 0.0,
        **_: object,
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        torch = _require_torch()
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=eta_min
        )


@SCHEDULERS.register("warm_restarts")
class _WarmRestartsBuilder:
    def __call__(
        self,
        optimizer: "torch.optim.Optimizer",
        epochs: int,
        t0: int = 10,
        t_mult: int = 2,
        eta_min: float = 0.0,
        **_: object,
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        torch = _require_torch()
        del epochs
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t0, T_mult=t_mult, eta_min=eta_min
        )


@SCHEDULERS.register("one_cycle")
class _OneCycleBuilder:
    def __call__(
        self,
        optimizer: "torch.optim.Optimizer",
        epochs: int,
        steps_per_epoch: int,
        max_lr: float | None = None,
        pct_start: float = 0.3,
        **_: object,
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        torch = _require_torch()
        lr = max_lr if max_lr is not None else optimizer.param_groups[0]["lr"]
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
        )


@SCHEDULERS.register("plateau")
class _PlateauBuilder:
    def __call__(
        self,
        optimizer: "torch.optim.Optimizer",
        epochs: int,
        mode: str = "min",
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-7,
        **_: object,
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        torch = _require_torch()
        del epochs
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )


@SCHEDULERS.register("linear_warmup")
class _LinearWarmupBuilder:
    """Linear warmup followed by cosine decay to ``eta_min``."""

    def __call__(
        self,
        optimizer: "torch.optim.Optimizer",
        epochs: int,
        warmup_epochs: int = 5,
        eta_min: float = 0.0,
        **_: object,
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        torch = _require_torch()

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = (epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            return eta_min + 0.5 * (1.0 - eta_min) * (1.0 + __import__("math").cos(
                __import__("math").pi * progress
            ))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_scheduler(
    name: str,
    optimizer: "torch.optim.Optimizer",
    epochs: int,
    steps_per_epoch: int = 1,
    **params: object,
) -> "torch.optim.lr_scheduler.LRScheduler":
    """Builds a scheduler from the registry.

    Args:
        name: Registered scheduler name.
        optimizer: The optimizer to schedule.
        epochs: Total training epochs.
        steps_per_epoch: Batches per epoch (for OneCycle).
        **params: Scheduler-specific hyperparameters.

    Returns:
        The constructed scheduler.
    """
    builder_cls = SCHEDULERS.get(name)
    builder = builder_cls() if isinstance(builder_cls, type) else builder_cls
    return builder(
        optimizer, epochs=epochs, steps_per_epoch=steps_per_epoch, **params
    )  # type: ignore[operator]
