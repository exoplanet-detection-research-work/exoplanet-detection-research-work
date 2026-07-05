"""Contrastive representation pretraining (Module 7)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from exodet.ml.device import select_device
from exodet.ml.models import BaseTorchModel
from exodet.representation.containers import RepresentationDataset
from exodet.utils.io import write_json

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["NTXentLoss", "ProjectionHead", "ContrastivePretrainer", "run_contrastive_pretraining"]

logger = logging.getLogger(__name__)


class ProjectionHead:
    """MLP projection for contrastive learning."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64) -> None:
        import torch

        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NTXentLoss:
    """Normalised temperature-scaled cross entropy (SimCLR)."""

    def __init__(self, temperature: float = 0.1) -> None:
        import torch

        self.temperature = temperature
        self._torch = torch

    def __call__(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        torch = self._torch
        import torch.nn.functional as F

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        batch = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        mask = torch.eye(2 * batch, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -1e9)
        targets = torch.cat(
            [torch.arange(batch, 2 * batch), torch.arange(0, batch)], dim=0
        ).to(z.device)
        return F.cross_entropy(sim, targets)


@dataclass
class ContrastivePretrainer:
    """SimCLR-style contrastive pretraining on augmented view pairs."""

    temperature: float = 0.1
    epochs: int = 10
    learning_rate: float = 1e-3
    projection_dim: int = 64

    def pretrain(
        self,
        model: BaseTorchModel,
        dataset: RepresentationDataset,
        checkpoint_dir: Path,
        batch_size: int = 32,
        augment_config: dict[str, Any] | None = None,
    ) -> Path:
        import torch

        from exodet.training.augmentation import (
            AugmentedTorchDataset,
            build_training_augmentation,
        )

        device = select_device("auto").device
        default_aug = {
            "enabled": True,
            "probability": 1.0,
            "steps": [{"name": "gaussian_noise", "params": {}}],
        }
        aug1 = build_training_augmentation(augment_config or default_aug)
        aug2_cfg = dict(augment_config or default_aug)
        aug2_cfg["steps"] = [{"name": "flux_scaling", "params": {}}]
        aug2 = build_training_augmentation(aug2_cfg)
        ds1 = AugmentedTorchDataset(dataset, aug1, seed=0)
        ds2 = AugmentedTorchDataset(dataset, aug2, seed=1)
        from torch.utils.data import DataLoader

        from exodet.ml.data import collate_ml_batch

        loader1 = DataLoader(
            ds1,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_ml_batch(b, use_views="both"),
        )
        loader2 = DataLoader(
            ds2,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_ml_batch(b, use_views="both"),
        )
        sample = next(iter(loader1))
        input_dim = sum(t.shape[1] for t in (sample.global_view, sample.local_view, sample.features) if t is not None)
        model._ensure_module(input_dim, device)
        network = model.module
        embed_dim = getattr(network.config, "embed_dim", 128)
        projector = ProjectionHead(embed_dim, out_dim=self.projection_dim).to(device)
        criterion = NTXentLoss(self.temperature)
        optimizer = torch.optim.AdamW(
            list(network.parameters()) + list(projector.parameters()),
            lr=self.learning_rate,
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.epochs + 1):
            total = 0.0
            n = 0
            network.train()
            for batch1, batch2 in zip(loader1, loader2, strict=False):
                global_view = batch1.global_view.to(device) if batch1.global_view is not None else None
                local_view = batch1.local_view.to(device) if batch1.local_view is not None else None
                features = batch1.features.to(device) if batch1.features is not None else None
                global_view2 = batch2.global_view.to(device) if batch2.global_view is not None else None
                local_view2 = batch2.local_view.to(device) if batch2.local_view is not None else None
                features2 = batch2.features.to(device) if batch2.features is not None else None
                out1 = network(global_view=global_view, local_view=local_view, physics=features)
                out2 = network(global_view=global_view2, local_view=local_view2, physics=features2)
                z1 = projector(out1.fused)
                z2 = projector(out2.fused)
                loss = criterion(z1, z2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
                n += 1
            logger.info("Contrastive epoch %d loss=%.4f", epoch, total / max(1, n))

        out_path = checkpoint_dir / "contrastive_encoder.pt"
        model.save(out_path)
        write_json({"temperature": self.temperature, "epochs": self.epochs}, checkpoint_dir / "contrastive_report.json")
        return out_path


def run_contrastive_pretraining(
    model: BaseTorchModel,
    dataset: RepresentationDataset,
    config: dict[str, Any],
    checkpoint_dir: Path,
) -> Path:
    """Runs contrastive pretraining from YAML config."""
    pretrainer = ContrastivePretrainer(
        temperature=float(config.get("temperature", 0.1)),
        epochs=int(config.get("epochs", 10)),
        learning_rate=float(config.get("learning_rate", 1e-3)),
        projection_dim=int(config.get("projection_dim", 64)),
    )
    return pretrainer.pretrain(
        model,
        dataset,
        checkpoint_dir,
        batch_size=int(config.get("batch_size", 32)),
        augment_config=config.get("augmentation"),
    )
