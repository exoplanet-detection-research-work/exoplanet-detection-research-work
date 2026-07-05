"""Model explainability: Grad-CAM, integrated gradients, attention maps."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exodet.ml.data import MlBatch
from exodet.ml.models import BaseTorchModel
from exodet.representation.containers import DatasetSample
from exodet.utils.io import ensure_dir

__all__ = ["ExplainabilityResult", "ExplainabilityEngine"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExplainabilityResult:
    """Paths to generated explainability artefacts."""

    grad_cam_path: str | None = None
    integrated_gradients_path: str | None = None
    attention_heatmap_path: str | None = None
    attention_rollout_path: str | None = None
    occlusion_path: str | None = None
    feature_attribution_path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "grad_cam_path": self.grad_cam_path,
            "integrated_gradients_path": self.integrated_gradients_path,
            "attention_heatmap_path": self.attention_heatmap_path,
            "attention_rollout_path": self.attention_rollout_path,
            "occlusion_path": self.occlusion_path,
            "feature_attribution_path": self.feature_attribution_path,
            "meta": dict(self.meta),
        }


class ExplainabilityEngine:
    """Generates publication-quality explainability figures."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.methods = tuple(cfg.get("methods", ["grad_cam", "integrated_gradients", "attention"]))
        self.dpi = int(cfg.get("dpi", 150))
        self.n_ig_steps = int(cfg.get("n_integrated_steps", 32))
        self.occlusion_window = int(cfg.get("occlusion_window", 5))

    def explain(
        self,
        model: BaseTorchModel,
        sample: DatasetSample,
        output_dir: Path,
        prefix: str,
    ) -> ExplainabilityResult:
        """Runs configured explainability methods."""
        if not self.enabled:
            return ExplainabilityResult()

        import torch

        ensure_dir(output_dir)
        device = model._device or torch.device("cpu")
        input_dim = (
            sample.global_view.size
            + sample.local_view.size
            + sample.features.size
        )
        model._ensure_module(input_dim, device)

        batch = MlBatch(
            global_view=torch.from_numpy(sample.global_view[None].astype(np.float32)).to(device),
            local_view=torch.from_numpy(sample.local_view[None].astype(np.float32)).to(device),
            features=torch.from_numpy(sample.features[None].astype(np.float32)).to(device),
            labels=torch.tensor([sample.label], dtype=torch.float32, device=device),
            weights=torch.tensor([sample.weight], dtype=torch.float32, device=device),
            sample_ids=(sample.sample_id,),
            target_ids=(sample.target_id,),
        )

        paths: dict[str, str | None] = {
            "grad_cam_path": None,
            "integrated_gradients_path": None,
            "attention_heatmap_path": None,
            "attention_rollout_path": None,
            "occlusion_path": None,
            "feature_attribution_path": None,
        }

        if "grad_cam" in self.methods:
            paths["grad_cam_path"] = self._grad_cam(model, batch, output_dir, prefix)

        if "integrated_gradients" in self.methods:
            paths["integrated_gradients_path"] = self._integrated_gradients(
                model, batch, output_dir, prefix
            )

        if "attention" in self.methods:
            paths["attention_heatmap_path"] = self._attention_heatmap(
                model, batch, output_dir, prefix
            )
            paths["attention_rollout_path"] = paths["attention_heatmap_path"]

        if "occlusion" in self.methods:
            paths["occlusion_path"] = self._occlusion_sensitivity(
                model, sample, output_dir, prefix
            )

        if "feature_attribution" in self.methods:
            paths["feature_attribution_path"] = self._feature_attribution(
                model, batch, output_dir, prefix
            )

        return ExplainabilityResult(**paths)

    def _grad_cam(
        self,
        model: BaseTorchModel,
        batch: MlBatch,
        output_dir: Path,
        prefix: str,
    ) -> str:
        import torch

        network = model.module
        network.eval()
        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def fwd_hook(_module: torch.nn.Module, _inp: tuple, out: torch.Tensor) -> None:
            activations.append(out.detach())

        def bwd_hook(_module: torch.nn.Module, _gin: tuple, grad_out: tuple) -> None:
            gradients.append(grad_out[0].detach())

        target_layer = None
        if hasattr(network, "local_cnn") and network.local_cnn is not None:
            target_layer = network.local_cnn.encoder[-1]
        if target_layer is None:
            return ""

        handle_f = target_layer.register_forward_hook(fwd_hook)
        handle_b = target_layer.register_full_backward_hook(bwd_hook)

        logits = model.forward_batch(batch)
        network.zero_grad(set_to_none=True)
        logits.sum().backward()

        handle_f.remove()
        handle_b.remove()
        if not activations or not gradients:
            return ""

        act = activations[0][0]
        grad = gradients[0][0]
        weights = grad.mean(dim=-1)
        cam = torch.relu((weights.unsqueeze(-1) * act).sum(dim=0))
        cam_np = cam.cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / max(cam_np.max() - cam_np.min(), 1e-9)

        path = output_dir / f"{prefix}_grad_cam.png"
        fig, ax = plt.subplots(figsize=(6, 3))
        if batch.local_view is not None:
            ax.plot(batch.local_view[0].cpu().numpy(), label="local view", color="0.3")
        ax.fill_between(np.arange(len(cam_np)), cam_np * cam_np.max(), alpha=0.4, color="crimson")
        ax.set_xlabel("Phase bin")
        ax.set_ylabel("Flux / attribution")
        ax.set_title("Grad-CAM (local CNN)")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return str(path)

    def _integrated_gradients(
        self,
        model: BaseTorchModel,
        batch: MlBatch,
        output_dir: Path,
        prefix: str,
    ) -> str:
        import torch

        network = model.module
        network.eval()
        parts: list[torch.Tensor] = []
        if batch.local_view is not None:
            parts.append(batch.local_view)
        if batch.global_view is not None:
            parts.append(batch.global_view)
        if not parts:
            return ""
        x = torch.cat(parts, dim=-1)
        baseline = torch.zeros_like(x)
        deltas = (x - baseline) / self.n_ig_steps
        accumulated = torch.zeros_like(x)
        for step in range(1, self.n_ig_steps + 1):
            scaled = baseline + deltas * step
            scaled = scaled.detach().requires_grad_(True)
            g_len = batch.global_view.shape[1] if batch.global_view is not None else 0
            l_len = batch.local_view.shape[1] if batch.local_view is not None else 0
            g_view = scaled[:, :g_len] if batch.global_view is not None else None
            l_view = scaled[:, g_len : g_len + l_len] if batch.local_view is not None else None
            mb = MlBatch(
                global_view=g_view,
                local_view=l_view,
                features=batch.features,
                labels=batch.labels,
                weights=batch.weights,
                sample_ids=batch.sample_ids,
                target_ids=batch.target_ids,
            )
            logits = model.forward_batch(mb)
            network.zero_grad(set_to_none=True)
            logits.sum().backward()
            accumulated += scaled.grad.detach()
        attributions = (x - baseline) * accumulated / self.n_ig_steps
        attr_np = attributions[0].cpu().numpy()

        path = output_dir / f"{prefix}_integrated_gradients.png"
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(np.arange(len(attr_np)), attr_np, width=1.0, color="steelblue")
        ax.set_xlabel("Input bin")
        ax.set_ylabel("Attribution")
        ax.set_title("Integrated Gradients")
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return str(path)

    def _attention_heatmap(
        self,
        model: BaseTorchModel,
        batch: MlBatch,
        output_dir: Path,
        prefix: str,
    ) -> str:
        import torch

        network = model.module
        network.eval()
        with torch.no_grad():
            out = network(
                global_view=batch.global_view,
                local_view=batch.local_view,
                physics=batch.features,
            )
        attn = out.cls_attention
        if attn is None:
            return ""
        weights = attn[0].mean(dim=0).cpu().numpy()

        path = output_dir / f"{prefix}_attention.png"
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(weights[None, :], aspect="auto", cmap="magma")
        ax.set_xlabel("Token")
        ax.set_yticks([])
        ax.set_title("Transformer attention (CLS rollout)")
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return str(path)

    def _occlusion_sensitivity(
        self,
        model: BaseTorchModel,
        sample: DatasetSample,
        output_dir: Path,
        prefix: str,
    ) -> str:
        import torch

        device = model._device or torch.device("cpu")
        local = sample.local_view.astype(np.float32)
        baseline_prob = float(
            model.predict_proba(
                np.concatenate(
                    [sample.global_view, local, sample.features], axis=0
                )[None]
            )[0]
        )
        scores = np.zeros(len(local))
        window = self.occlusion_window
        for i in range(len(local)):
            masked = local.copy()
            lo = max(0, i - window // 2)
            hi = min(len(local), i + window // 2 + 1)
            masked[lo:hi] = np.median(local)
            prob = float(
                model.predict_proba(
                    np.concatenate(
                        [sample.global_view, masked, sample.features], axis=0
                    )[None]
                )[0]
            )
            scores[i] = abs(prob - baseline_prob)

        path = output_dir / f"{prefix}_occlusion.png"
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(scores, color="darkorange")
        ax.set_xlabel("Phase bin")
        ax.set_ylabel("|ΔP(planet)|")
        ax.set_title("Occlusion sensitivity")
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return str(path)

    def _feature_attribution(
        self,
        model: BaseTorchModel,
        batch: MlBatch,
        output_dir: Path,
        prefix: str,
    ) -> str:
        import torch

        if batch.features is None:
            return ""
        features = batch.features.detach().clone().requires_grad_(True)
        mb = MlBatch(
            global_view=batch.global_view,
            local_view=batch.local_view,
            features=features,
            labels=batch.labels,
            weights=batch.weights,
            sample_ids=batch.sample_ids,
            target_ids=batch.target_ids,
        )
        logits = model.forward_batch(mb)
        model.module.zero_grad(set_to_none=True)
        logits.sum().backward()
        attr = features.grad[0].cpu().numpy()

        path = output_dir / f"{prefix}_feature_attribution.png"
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(np.arange(len(attr)), attr, color="seagreen")
        ax.set_xlabel("Physics feature")
        ax.set_ylabel("Gradient")
        ax.set_title("Feature attribution")
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return str(path)
