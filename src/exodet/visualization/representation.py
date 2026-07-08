"""Diagnostic figures of the representation stage (Module 11).

All figures follow the project publication style and export as both
PDF and PNG via the shared ``save_figure`` helper.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from exodet.data.base import LightCurve
from exodet.representation.containers import (
    DatasetSample,
    PhaseFoldedCurve,
    View,
)
from exodet.tce.candidate import TransitCandidate
from exodet.utils.paths import safe_filename
from exodet.visualization.style import apply_publication_style, save_figure

if TYPE_CHECKING:  # pragma: no cover - typing only
    from exodet.representation.pipeline import RepresentationPipeline

__all__ = [
    "plot_phase_folded",
    "plot_view",
    "plot_alignment",
    "plot_interpolation_diagnostics",
    "plot_feature_distributions",
    "plot_normalization_diagnostics",
    "generate_representation_figures",
]

logger = logging.getLogger(__name__)

_POINT_STYLE = {"s": 2.0, "color": "0.4", "alpha": 0.5, "linewidths": 0.0}


def plot_phase_folded(folded: PhaseFoldedCurve) -> plt.Figure:
    """Plots the phase-folded light curve with the transit window.

    Args:
        folded: The folded curve.

    Returns:
        The created figure.
    """
    apply_publication_style()
    figure, axis = plt.subplots()
    axis.scatter(folded.phase, folded.flux, **_POINT_STYLE)
    half = 0.5 * folded.duty_cycle
    axis.axvspan(-half, half, color="C3", alpha=0.15, label="transit window")
    axis.set_xlabel("Phase")
    axis.set_ylabel("Relative flux")
    axis.set_title(
        f"{folded.candidate_id} — folded at P={folded.period_days:.4f} d"
    )
    axis.legend(loc="lower right")
    return figure


def plot_view(view: View, candidate_id: str) -> plt.Figure:
    """Plots a binned view with interpolated bins highlighted.

    Args:
        view: The global or local view.
        candidate_id: Candidate identifier for the title.

    Returns:
        The created figure.
    """
    apply_publication_style()
    figure, axis = plt.subplots()
    axis.plot(view.bin_centers, view.values, color="0.2", linewidth=0.9)
    axis.set_xlabel("Phase")
    axis.set_ylabel("Normalized flux")
    axis.set_title(
        f"{candidate_id} — {view.kind} view ({view.n_bins} bins, "
        f"{view.empty_fraction:.1%} interpolated)"
    )
    return figure


def plot_alignment(
    folded_raw: PhaseFoldedCurve, folded_aligned: PhaseFoldedCurve
) -> plt.Figure:
    """Compares the transit region before and after alignment.

    Args:
        folded_raw: Folded curve without epoch correction.
        folded_aligned: Folded curve with alignment applied.

    Returns:
        The created figure.
    """
    apply_publication_style()
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    window = 2.5 * folded_aligned.duty_cycle
    for axis, folded, title in (
        (left, folded_raw, "before alignment"),
        (right, folded_aligned, "after alignment"),
    ):
        zoom = np.abs(folded.phase) < window
        axis.scatter(folded.phase[zoom], folded.flux[zoom], **_POINT_STYLE)
        axis.axvline(0.0, color="C3", linewidth=1.0, alpha=0.8)
        axis.set_xlabel("Phase")
        axis.set_title(title)
    left.set_ylabel("Relative flux")
    figure.suptitle(
        f"{folded_aligned.candidate_id} — epoch correction "
        f"{folded_aligned.epoch_correction_days * 24 * 60:+.2f} min"
    )
    return figure


def plot_interpolation_diagnostics(view: View, candidate_id: str) -> plt.Figure:
    """Shows the view with per-bin occupancy to judge interpolation.

    Args:
        view: The view to diagnose.
        candidate_id: Candidate identifier for the title.

    Returns:
        The created figure.
    """
    apply_publication_style()
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.0, 5.5), sharex=True, height_ratios=[3, 1]
    )
    top.plot(view.bin_centers, view.values, color="0.2", linewidth=0.9)
    top.set_ylabel("Normalized flux")
    top.set_title(
        f"{candidate_id} — {view.kind} view interpolation "
        f"({view.interpolation}, {view.n_empty_bins} empty bins)"
    )
    counts = view.meta.get("median_samples_per_bin")
    bottom.bar(
        view.bin_centers,
        np.where(np.isfinite(view.values), 1.0, 0.0),
        width=view.bin_centers[1] - view.bin_centers[0],
        color="0.6",
    )
    bottom.set_xlabel("Phase")
    bottom.set_ylabel("occupied")
    if counts is not None:
        bottom.annotate(
            f"median samples/bin: {counts:.1f}",
            xy=(0.02, 0.7),
            xycoords="axes fraction",
            fontsize=8,
        )
    return figure


def plot_feature_distributions(
    samples: list[DatasetSample], directory: Path | str, max_features: int = 16
) -> list[Path]:
    """Plots histograms of the leading physics features.

    Args:
        samples: Dataset samples.
        directory: Output directory.
        max_features: Maximum number of feature panels.

    Returns:
        The written figure paths.
    """
    if not samples:
        return []
    apply_publication_style()
    names = samples[0].feature_names[:max_features]
    matrix = np.stack([s.features for s in samples])[:, : len(names)]
    n_cols = 4
    n_rows = int(np.ceil(len(names) / n_cols))
    figure, axes = plt.subplots(
        n_rows, n_cols, figsize=(11.0, 2.4 * n_rows), squeeze=False
    )
    for index, name in enumerate(names):
        axis = axes[index // n_cols][index % n_cols]
        values = matrix[:, index]
        values = values[np.isfinite(values)]
        if values.size:
            axis.hist(values, bins=min(30, max(5, values.size // 2)), color="0.5")
        axis.set_title(name, fontsize=8)
        axis.tick_params(labelsize=7)
    for index in range(len(names), n_rows * n_cols):
        axes[index // n_cols][index % n_cols].axis("off")
    figure.suptitle(f"Physics feature distributions ({len(samples)} samples)")
    figure.tight_layout()
    paths = save_figure(figure, directory, "dataset_feature_distributions")
    plt.close(figure)
    return paths


def plot_normalization_diagnostics(
    raw_features: np.ndarray,
    scaled_features: np.ndarray,
    names: tuple[str, ...],
    directory: Path | str,
    max_features: int = 8,
) -> list[Path]:
    """Compares raw and scaled feature distributions side by side.

    Args:
        raw_features: Unscaled feature matrix ``(n, f)``.
        scaled_features: Scaled feature matrix ``(n, f)``.
        names: Feature names.
        directory: Output directory.
        max_features: Number of features shown.

    Returns:
        The written figure paths.
    """
    apply_publication_style()
    n = min(max_features, len(names))
    figure, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6), squeeze=False)
    for index in range(n):
        for row, matrix, label in ((0, raw_features, "raw"), (1, scaled_features, "scaled")):
            axis = axes[row][index]
            values = matrix[:, index]
            values = values[np.isfinite(values)]
            if values.size:
                axis.hist(values, bins=min(25, max(5, values.size // 2)), color="0.5")
            if row == 0:
                axis.set_title(names[index], fontsize=8)
            if index == 0:
                axis.set_ylabel(label)
            axis.tick_params(labelsize=7)
    figure.suptitle("Feature normalization diagnostics")
    figure.tight_layout()
    paths = save_figure(figure, directory, "dataset_normalization_diagnostics")
    plt.close(figure)
    return paths


def generate_representation_figures(
    curve: LightCurve,
    candidate: TransitCandidate,
    pipeline: "RepresentationPipeline",
    sample: DatasetSample,
    directory: Path | str,
) -> list[Path]:
    """Exports the per-candidate diagnostic figure set.

    Regenerates the folded curve and views (cheap) so figures work even
    when the sample itself came from the cache.

    Args:
        curve: The processed light curve.
        candidate: The represented candidate.
        pipeline: The configured representation pipeline.
        sample: The built sample (for titles/metadata).
        directory: Output directory.

    Returns:
        The written figure paths.
    """
    directory = Path(directory)
    slug = safe_filename(sample.sample_id).lower()

    folded = pipeline.folder.fold(curve, candidate)
    global_view = pipeline.global_generator.generate(folded)
    local_view = pipeline.local_generator.generate(folded)

    raw_folder = type(pipeline.folder)(align=False)
    folded_raw = raw_folder.fold(curve, candidate)

    figures = [
        (f"{slug}_phase_folded", plot_phase_folded(folded)),
        (f"{slug}_global_view", plot_view(global_view, candidate.candidate_id)),
        (f"{slug}_local_view", plot_view(local_view, candidate.candidate_id)),
        (f"{slug}_alignment", plot_alignment(folded_raw, folded)),
        (
            f"{slug}_interpolation",
            plot_interpolation_diagnostics(local_view, candidate.candidate_id),
        ),
    ]
    written: list[Path] = []
    for name, figure in figures:
        written.extend(save_figure(figure, directory, name))
        plt.close(figure)
    logger.info(
        "Sample %s: wrote %d representation figure file(s).",
        sample.sample_id,
        len(written),
    )
    return written
