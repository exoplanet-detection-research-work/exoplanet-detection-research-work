"""Self-supervised masked time-series pretraining (Module 6)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from exodet.ml.data import MlBatch
from exodet.ml.device import select_device
from exodet.ml.models import BaseTorchModel
from exodet.representation.containers import RepresentationDataset
from exodet.utils.io import write_json

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["MaskedReconstructionHead", "MaskedPretrainer", "run_masked_pretraining"]

logger = logging.getLogger(__name__)


class MaskedReconstructionHead(torch.nn.Module):
    """Linear decoder reconstructing masked view bins."""

    def __init__(self, embed_dim: int, output_dim: int) -> None:
        import torch

        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(embed_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


@dataclass
class MaskedPretrainer:
    """Masked patch modeling on global/local views.

    Attributes:
        mask_fraction: Fraction of bins to mask per sample.
        patch_size: Contiguous patch length for patch masking.
        strategy: ``random`` or ``patch``.
        epochs: Pretraining epochs.
        learning_rate: AdamW learning rate.
    """

    mask_fraction: float = 0.15
    patch_size: int = 16
    strategy: str = "random"
    epochs: int = 10
    learning_rate: float = 1e-3

    def _mask_indices(self, length: int, rng: np.random.Generator) -> np.ndarray:
        n_mask = max(1, int(length * self.mask_fraction))
        if self.strategy == "patch":
            start = int(rng.integers(0, max(1, length - self.patch_size)))
            indices = np.arange(start, min(start + self.patch_size, length))
        else:
            indices = rng.choice(length, size=n_mask, replace=False)
        mask = np.zeros(length, dtype=bool)
        mask[indices] = True
        return mask

    def pretrain(
        self,
        model: BaseTorchModel,
        dataset: RepresentationDataset,
        checkpoint_dir: Path,
        batch_size: int = 32,
    ) -> Path:
        """Runs encoder pretraining and exports checkpoint.

        Args:
            model: Student architecture (encoder branches used).
            dataset: Unlabeled or labeled representation data.
            checkpoint_dir: Output directory.
            batch_size: Mini-batch size.

        Returns:
            Path to exported encoder checkpoint.
        """
        import torch

        from exodet.ml.data import RepresentationDataModule

        device_info = select_device("auto")
        device = device_info.device
        dm = RepresentationDataModule(train=dataset, batch_size=batch_size)
        loader = dm.train_dataloader()
        sample = next(iter(loader))
        input_dim = sum(
            t.shape[1] for t in (sample.global_view, sample.local_view, sample.features) if t is not None
        )
        model._ensure_module(input_dim, device)
        network = model.module
        embed_dim = getattr(network.config, "embed_dim", 128)
        recon = MaskedReconstructionHead(embed_dim, input_dim).to(device)
        optimizer = torch.optim.AdamW(
            list(network.parameters()) + list(recon.parameters()),
            lr=self.learning_rate,
        )
        rng = np.random.default_rng(0)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            n_batches = 0
            network.train()
            for batch in loader:
                batch = MlBatch(
                    global_view=batch.global_view.to(device) if batch.global_view is not None else None,
                    local_view=batch.local_view.to(device) if batch.local_view is not None else None,
                    features=batch.features.to(device) if batch.features is not None else None,
                    labels=batch.labels,
                    weights=batch.weights,
                    sample_ids=batch.sample_ids,
                    target_ids=batch.target_ids,
                )
                parts = []
                if batch.global_view is not None:
                    parts.append(batch.global_view)
                if batch.local_view is not None:
                    parts.append(batch.local_view)
                if batch.features is not None:
                    parts.append(batch.features)
                flat = torch.cat(parts, dim=1)
                mask = torch.zeros_like(flat, dtype=torch.bool)
                for row in range(flat.shape[0]):
                    idx = self._mask_indices(flat.shape[1], rng)
                    mask[row, idx] = True
                masked = flat.clone()
                masked[mask] = 0.0
                g_len = batch.global_view.shape[1] if batch.global_view is not None else 0
                l_len = batch.local_view.shape[1] if batch.local_view is not None else 0
                g_view = masked[:, :g_len] if batch.global_view is not None else None
                l_view = masked[:, g_len : g_len + l_len] if batch.local_view is not None else None
                f_view = masked[:, g_len + l_len :] if batch.features is not None else None
                out = network(
                    global_view=g_view,
                    local_view=l_view,
                    physics=f_view,
                )
                pred = recon(out.fused)
                loss = ((pred[mask] - flat[mask]) ** 2).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())
                n_batches += 1
            logger.info("Pretrain epoch %d loss=%.4f", epoch, total_loss / max(1, n_batches))

        out_path = checkpoint_dir / "pretrained_encoder.pt"
        model.save(out_path)
        write_json({"epochs": self.epochs, "mask_fraction": self.mask_fraction}, checkpoint_dir / "pretrain_report.json")
        return out_path


def run_masked_pretraining(
    model: BaseTorchModel,
    dataset: RepresentationDataset,
    config: dict[str, Any],
    checkpoint_dir: Path,
) -> Path:
    """Convenience entrypoint from YAML config."""
    trainer = MaskedPretrainer(
        mask_fraction=float(config.get("mask_fraction", 0.15)),
        patch_size=int(config.get("patch_size", 16)),
        strategy=str(config.get("strategy", "random")),
        epochs=int(config.get("epochs", 10)),
        learning_rate=float(config.get("learning_rate", 1e-3)),
    )
    return trainer.pretrain(
        model, dataset, checkpoint_dir, batch_size=int(config.get("batch_size", 32))
    )
