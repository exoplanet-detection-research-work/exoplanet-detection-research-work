"""ML training settings parsed from the experiment YAML.

Extended hyperparameters (loss, optimizer, scheduler, AMP, callbacks,
checkpointing, tracking, cross-validation) live in
``training.trainer.params`` so the existing :class:`TrainingConfig`
schema stays unchanged. :func:`load_ml_settings` merges the top-level
``training`` block with the trainer params dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from exodet.config.schema import ComponentConfig, TrainingConfig
from exodet.exceptions import ConfigurationError

__all__ = ["MlSettings", "load_ml_settings"]


@dataclass(frozen=True, slots=True)
class MlSettings:
    """Resolved machine-learning training settings.

    Attributes:
        loss: Loss function component.
        optimizer: Optimizer component.
        scheduler: LR scheduler component (optional).
        amp: Mixed-precision mode (``none``, ``fp16``, ``bf16``).
        grad_clip_norm: Max gradient norm; ``0`` disables clipping.
        checkpoint: Checkpoint policy dict.
        callbacks: Ordered callback component list.
        tracking: Experiment-tracking backend settings.
        cross_validation: CV settings (disabled when ``enabled`` is false).
        inference: Inference defaults (batch size, calibration hooks).
        num_workers: DataLoader worker count.
        pin_memory: Whether to pin host memory for GPU transfer.
        use_views: Input channels (``global``, ``local``, ``both``, ``features_only``).
        backend: ``torch`` or ``sklearn``.
    """

    loss: ComponentConfig
    optimizer: ComponentConfig
    scheduler: ComponentConfig | None
    amp: str = "none"
    grad_clip_norm: float = 0.0
    checkpoint: dict[str, Any] = field(default_factory=dict)
    callbacks: tuple[ComponentConfig, ...] = ()
    tracking: dict[str, Any] = field(default_factory=dict)
    cross_validation: dict[str, Any] = field(default_factory=dict)
    inference: dict[str, Any] = field(default_factory=dict)
    num_workers: int = 0
    pin_memory: bool = True
    use_views: str = "both"
    backend: str = "torch"

    _AMP = frozenset({"none", "fp16", "bf16"})
    _VIEWS = frozenset({"global", "local", "both", "features_only"})
    _BACKENDS = frozenset({"torch", "sklearn"})

    @classmethod
    def from_training_config(cls, config: TrainingConfig) -> "MlSettings":
        """Builds ML settings from a :class:`TrainingConfig`.

        Args:
            config: The experiment training section.

        Returns:
            Parsed ML settings.

        Raises:
            ConfigurationError: If params are invalid.
        """
        params = dict(config.trainer.params)
        loss_raw = params.pop("loss", {"name": "bce", "params": {}})
        opt_raw = params.pop("optimizer", {"name": "adamw", "params": {}})
        sched_raw = params.pop("scheduler", None)

        amp = str(params.pop("amp", "none")).lower()
        if amp not in cls._AMP:
            raise ConfigurationError(
                f"training.trainer.params.amp must be one of {sorted(cls._AMP)}, "
                f"got '{amp}'."
            )
        grad_clip = float(params.pop("grad_clip_norm", 0.0))
        if grad_clip < 0:
            raise ConfigurationError("grad_clip_norm must be >= 0.")

        use_views = str(params.pop("use_views", "both"))
        if use_views not in cls._VIEWS:
            raise ConfigurationError(
                f"use_views must be one of {sorted(cls._VIEWS)}, got '{use_views}'."
            )
        backend = str(params.pop("backend", "torch"))
        if backend not in cls._BACKENDS:
            raise ConfigurationError(
                f"backend must be one of {sorted(cls._BACKENDS)}, got '{backend}'."
            )

        callbacks_raw = params.pop("callbacks", [])
        if not isinstance(callbacks_raw, list):
            raise ConfigurationError("callbacks must be a list.")

        return cls(
            loss=ComponentConfig.from_dict(loss_raw, "training.trainer.params.loss"),
            optimizer=ComponentConfig.from_dict(
                opt_raw, "training.trainer.params.optimizer"
            ),
            scheduler=(
                ComponentConfig.from_dict(sched_raw, "training.trainer.params.scheduler")
                if sched_raw is not None
                else None
            ),
            amp=amp,
            grad_clip_norm=grad_clip,
            checkpoint=dict(params.pop("checkpoint", {})),
            callbacks=tuple(
                ComponentConfig.from_dict(c, f"training.trainer.params.callbacks[{i}]")
                for i, c in enumerate(callbacks_raw)
            ),
            tracking=dict(params.pop("tracking", {})),
            cross_validation=dict(params.pop("cross_validation", {})),
            inference=dict(params.pop("inference", {})),
            num_workers=int(params.pop("num_workers", 0)),
            pin_memory=bool(params.pop("pin_memory", True)),
            use_views=use_views,
            backend=backend,
        )


def load_ml_settings(config: TrainingConfig) -> MlSettings:
    """Convenience wrapper around :meth:`MlSettings.from_training_config`.

    Args:
        config: Experiment training configuration.

    Returns:
        Parsed ML settings.
    """
    return MlSettings.from_training_config(config)
