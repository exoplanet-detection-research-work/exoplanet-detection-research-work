"""Training resume configuration for incremental updates."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from exodet.config.schema import ComponentConfig, ExperimentConfig
from exodet.update.checkpoint_manager import CheckpointDiscovery

__all__ = [
    "apply_training_resume",
    "checkpoint_extra_state",
]


def apply_training_resume(
    experiment: ExperimentConfig,
    discovery: CheckpointDiscovery,
    *,
    fresh_start: bool = False,
) -> ExperimentConfig:
    """Return experiment config with resume parameters set for ``run_training``."""
    if fresh_start:
        return experiment

    trainer_params = dict(experiment.training.trainer.params)
    trainer_params["auto_resume"] = False
    trainer_params["resume_from"] = str(discovery.path)

    training = replace(
        experiment.training,
        trainer=ComponentConfig(
            name=experiment.training.trainer.name,
            params=trainer_params,
        ),
    )
    return replace(experiment, training=training)


def checkpoint_extra_state(path: Path) -> dict[str, Any]:
    """Extract EMA/SWA/curriculum/extra state from a checkpoint payload."""
    try:
        import torch
    except ImportError:
        return {}
    if not path.is_file():
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    extra = dict(payload.get("extra", {}))
    return {
        "epoch": payload.get("epoch"),
        "metrics": payload.get("metrics", {}),
        "config_snapshot": payload.get("config_snapshot", {}),
        "extra": extra,
        "has_optimizer": bool(payload.get("optimizer_state")),
        "has_scheduler": bool(payload.get("scheduler_state")),
        "has_scaler": bool(payload.get("scaler_state")),
        "ema": extra.get("ema"),
        "swa": extra.get("swa"),
        "curriculum": extra.get("curriculum"),
        "global_step": extra.get("global_step"),
    }
