"""Shared helpers and test-only models for ML infrastructure tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from exodet.ml.data import MlBatch
from exodet.ml.models import BaseTorchModel
from exodet.models.base import MODELS
from exodet.representation.containers import DatasetSample, RepresentationDataset
from exodet.tce.candidate import TransitCandidate
from tests.test_tce import make_candidate

if TYPE_CHECKING:  # pragma: no cover
    import torch


@MODELS.register("linear_probe")
class LinearProbeModel(BaseTorchModel):
    """Minimal linear classifier for trainer infrastructure tests."""

    architecture_kind = "custom"

    def build_network(self, input_dim: int) -> "torch.nn.Module":
        import torch

        return torch.nn.Linear(input_dim, 1)

    def forward_batch(self, batch: MlBatch) -> "torch.Tensor":
        import torch

        parts: list[torch.Tensor] = []
        if batch.global_view is not None:
            parts.append(batch.global_view)
        if batch.local_view is not None:
            parts.append(batch.local_view)
        if batch.features is not None:
            parts.append(batch.features)
        features = torch.cat(parts, dim=1)
        return self.module(features).squeeze(-1)


def make_labeled_sample(
    seed: int = 0,
    label: int = 1,
    n_global: int = 32,
    n_local: int = 16,
    n_features: int = 8,
    target_id: str | None = None,
) -> DatasetSample:
    """Builds a synthetic labeled :class:`DatasetSample`."""
    rng = np.random.default_rng(seed)
    target_id = target_id or f"TIC {9000 + seed}"
    candidate = make_candidate(
        candidate_id=f"{target_id.replace(' ', '_')}-01",
        target_id=target_id,
        period_days=2.5,
        epoch_days=0.5,
        duration_days=0.1,
        depth=0.004,
    )
    names = tuple(f"f{i}" for i in range(n_features))
    return DatasetSample(
        sample_id=f"sample-{seed}",
        target_id=target_id,
        candidate=candidate,
        global_view=rng.normal(size=n_global),
        local_view=rng.normal(size=n_local),
        feature_names=names,
        features=rng.normal(size=n_features),
        label=label,
        weight=1.0,
    )


def make_representation_dataset(
    n_samples: int = 40,
    n_stars: int = 8,
    seed: int = 0,
    n_global: int = 32,
    n_local: int = 16,
    n_features: int = 8,
) -> RepresentationDataset:
    """Builds a synthetic labeled dataset with star-level groups."""
    samples = []
    for index in range(n_samples):
        star = index % n_stars
        label = index % 2
        samples.append(
            make_labeled_sample(
                seed=seed + index,
                label=label,
                n_global=n_global,
                n_local=n_local,
                n_features=n_features,
                target_id=f"TIC {9000 + star}",
            )
        )
    return RepresentationDataset(samples, version="test")


def fast_training_config(**overrides: Any) -> dict[str, Any]:
    """Minimal training YAML dict for tests."""
    raw: dict[str, Any] = {
        "trainer": {
            "name": "supervised",
            "params": {
                "backend": "torch",
                "use_views": "both",
                "loss": {"name": "bce", "params": {}},
                "optimizer": {"name": "adamw", "params": {}},
                "scheduler": {"name": "cosine", "params": {}},
                "amp": "none",
                "grad_clip_norm": 1.0,
                "num_workers": 0,
                "checkpoint": {
                    "monitor": "val_loss",
                    "mode": "min",
                    "save_best": True,
                    "save_last": True,
                    "top_k": 2,
                },
                "callbacks": [
                    {"name": "early_stopping", "params": {"patience": 3}},
                    {"name": "checkpoint", "params": {}},
                ],
                "tracking": {"enabled": True, "backends": ["csv"]},
                "cross_validation": {"enabled": False},
            },
        },
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 1e-2,
        "early_stopping_patience": 5,
    }
    if overrides:
        for key, value in overrides.items():
            if key == "trainer_params":
                raw["trainer"]["params"].update(value)
            else:
                raw[key] = value
    return raw
