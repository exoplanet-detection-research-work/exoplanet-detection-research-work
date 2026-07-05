"""Model interpretability visualizations (Module 10)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from exodet.models.classifier import ForwardOutput, HybridExoplanetNetwork
from exodet.visualization.style import apply_publication_style, save_figure

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = [
    "plot_cls_attention",
    "plot_cnn_activation_map",
    "plot_embedding_projection",
    "plot_feature_importance",
    "export_model_figures",
]

logger = logging.getLogger(__name__)


def plot_cls_attention(
    attention: "torch.Tensor | np.ndarray",
    global_bins: int,
    *,
    ax: plt.Axes | None = None,
    title: str = "CLS self-attention over global phase bins",
) -> plt.Axes:
    """Plots CLS-token attention weights across the global view."""
    apply_publication_style()
    weights = attention.detach().cpu().numpy() if hasattr(attention, "detach") else attention
    if weights.ndim == 2:
        weights = weights[0]
    # Index 0 is CLS; remaining indices align with global bins.
    curve_attn = weights[1 : 1 + global_bins]
    phases = np.linspace(-0.5, 0.5, len(curve_attn), endpoint=False)
    axis = ax or plt.subplots(figsize=(7, 3))[1]
    axis.plot(phases, curve_attn, color="steelblue")
    axis.axvline(0.0, color="crimson", linestyle="--", linewidth=0.9, label="transit")
    axis.set_xlabel("Orbital phase")
    axis.set_ylabel("Attention weight")
    axis.set_title(title)
    axis.legend(loc="upper right")
    return axis


def plot_cnn_activation_map(
    activations: "torch.Tensor | np.ndarray",
    local_bins: int,
    *,
    ax: plt.Axes | None = None,
    title: str = "CNN activation map (channel mean)",
) -> plt.Axes:
    """Plots mean CNN channel activations along the local view."""
    apply_publication_style()
    acts = activations.detach().cpu().numpy() if hasattr(activations, "detach") else activations
    if acts.ndim == 3:
        acts = acts[0].mean(axis=0)
    elif acts.ndim == 2:
        acts = acts.mean(axis=0)
    phases = np.linspace(-0.5, 0.5, min(len(acts), local_bins), endpoint=False)
    axis = ax or plt.subplots(figsize=(7, 3))[1]
    axis.plot(phases[: len(acts)], acts[: len(phases)], color="darkorange")
    axis.axvline(0.0, color="crimson", linestyle="--", linewidth=0.9)
    axis.set_xlabel("Orbital phase")
    axis.set_ylabel("Mean activation")
    axis.set_title(title)
    return axis


def plot_embedding_projection(
    embeddings: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    method: str = "pca",
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Projects fused embeddings to 2-D via PCA or t-SNE."""
    apply_publication_style()
    from sklearn.decomposition import PCA

    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("embeddings must be (n_samples, n_features) with n_samples >= 2.")

    if method == "tsne":
        from sklearn.manifold import TSNE

        projector: Any = TSNE(n_components=2, perplexity=min(30, embeddings.shape[0] - 1), random_state=0)
        title = title or "t-SNE of fused embeddings"
    else:
        projector = PCA(n_components=2, random_state=0)
        title = title or "PCA of fused embeddings"

    coords = projector.fit_transform(embeddings)
    axis = ax or plt.subplots(figsize=(6, 5))[1]
    if labels is None:
        axis.scatter(coords[:, 0], coords[:, 1], s=18, alpha=0.75)
    else:
        for label in np.unique(labels):
            mask = labels == label
            axis.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=18,
                alpha=0.75,
                label=str(label),
            )
        axis.legend(title="class", fontsize=9)
    axis.set_xlabel("component 1")
    axis.set_ylabel("component 2")
    axis.set_title(title)
    return axis


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: tuple[str, ...] | list[str],
    *,
    top_k: int = 20,
    ax: plt.Axes | None = None,
    title: str = "Physics feature importance",
) -> plt.Axes:
    """Bar plot of ranked physics feature importances."""
    apply_publication_style()
    order = np.argsort(np.abs(importances))[::-1][:top_k]
    names = [feature_names[i] for i in order]
    values = importances[order]
    axis = ax or plt.subplots(figsize=(7, 5))[1]
    axis.barh(range(len(names)), values, color="seagreen")
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels(names)
    axis.invert_yaxis()
    axis.set_xlabel("Importance")
    axis.set_title(title)
    return axis


def export_model_figures(
    output: ForwardOutput,
    figure_dir: Path | str,
    *,
    global_bins: int = 2001,
    local_bins: int = 401,
    prefix: str = "model",
) -> list[Path]:
    """Exports diagnostic figures for one forward pass.

    Args:
        output: Cached forward output from the network.
        figure_dir: Destination directory.
        global_bins: Global view bin count (for axis scaling).
        local_bins: Local view bin count.
        prefix: Filename stem prefix.

    Returns:
        List of written figure paths.
    """
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if output.cls_attention is not None:
        fig, _ = plt.subplots(figsize=(7, 3))
        plot_cls_attention(output.cls_attention, global_bins)
        written.extend(save_figure(fig, figure_dir, f"{prefix}_cls_attention"))
        plt.close(fig)

    if output.cnn_activations is not None:
        fig, _ = plt.subplots(figsize=(7, 3))
        plot_cnn_activation_map(output.cnn_activations, local_bins)
        written.extend(save_figure(fig, figure_dir, f"{prefix}_cnn_activation"))
        plt.close(fig)

    if output.fusion_attention is not None:
        weights = output.fusion_attention.detach().cpu().numpy()
        if weights.ndim == 2:
            weights = weights[0]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(range(len(weights)), weights, color="slateblue")
        ax.set_xlabel("Branch index")
        ax.set_ylabel("Fusion weight")
        ax.set_title("Cross-attention fusion weights")
        written.extend(save_figure(fig, figure_dir, f"{prefix}_fusion_attention"))
        plt.close(fig)

    logger.info("Exported %d model figure(s) to %s", len(written), figure_dir)
    return written
