"""Diagnostic figures for the preprocessing pipeline.

Each function returns a matplotlib ``Figure`` for programmatic use;
:func:`generate_preprocessing_figures` inspects the metadata produced
by the pipeline and exports every figure that is applicable, using the
project-wide publication style.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exodet.data.base import LightCurve
from exodet.preprocessing.normalization import Normalizer
from exodet.visualization.style import apply_publication_style, save_figure

__all__ = [
    "plot_raw_light_curve",
    "plot_detrended_light_curve",
    "plot_clipped_points",
    "plot_normalization_comparison",
    "generate_preprocessing_figures",
]

logger = logging.getLogger(__name__)

_RAW_STYLE = {"s": 2.0, "color": "0.3", "alpha": 0.6, "linewidths": 0.0}


def _denormalized(curve: LightCurve) -> LightCurve:
    """Inverts a recorded normalization to recover the input flux.

    Args:
        curve: A curve that may carry ``meta["normalization"]`` written
            by :class:`~exodet.preprocessing.normalization.Normalizer`.

    Returns:
        A copy with the normalization undone, or the curve unchanged
        when no normalization was recorded.
    """
    record = curve.meta.get("normalization")
    if not isinstance(record, dict):
        return curve
    stats = record["stats"]
    method = record["method"]
    if method == "minmax":
        flux = curve.flux * (stats["max"] - stats["min"]) + stats["min"]
    elif method == "median":
        flux = curve.flux * stats["median"]
    else:
        flux = curve.flux * stats["std"] + stats["mean"]
    return curve.replace_flux(flux, step_name="denormalize(figure)")


def plot_raw_light_curve(curve: LightCurve) -> plt.Figure:
    """Plots a light curve as delivered by the data source.

    Args:
        curve: The raw light curve.

    Returns:
        The created figure.
    """
    apply_publication_style()
    figure, axis = plt.subplots()
    axis.scatter(curve.time, curve.flux, **_RAW_STYLE)
    axis.set_xlabel("Time [d]")
    axis.set_ylabel("Flux")
    axis.set_title(f"{curve.target_id} — raw light curve")
    return figure


def plot_detrended_light_curve(
    raw: LightCurve, detrended: LightCurve
) -> plt.Figure:
    """Plots the raw flux with the fitted trend and the detrended result.

    Args:
        raw: The curve before detrending.
        detrended: The pipeline output carrying ``meta["trend"]``
            aligned with its cadences.

    Returns:
        The created two-panel figure.
    """
    apply_publication_style()
    figure, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(7.0, 6.0))

    top.scatter(raw.time, raw.flux, label="raw", **_RAW_STYLE)
    trend = detrended.meta.get("trend")
    if isinstance(trend, np.ndarray) and trend.shape == detrended.time.shape:
        stitched_medians = detrended.meta.get("sector_medians")
        # The trend lives in stitched (relative) units; project it back
        # to raw units when a single global median was applied.
        if isinstance(stitched_medians, dict) and len(stitched_medians) == 1:
            trend = trend * next(iter(stitched_medians.values()))
        top.plot(detrended.time, trend, color="C1", label="trend")
    top.set_ylabel("Flux")
    top.legend(loc="best")
    top.set_title(f"{raw.target_id} — variability removal")

    bottom.scatter(detrended.time, detrended.flux, **_RAW_STYLE)
    bottom.set_xlabel("Time [d]")
    bottom.set_ylabel("Relative flux")
    return figure


def plot_clipped_points(curve: LightCurve) -> plt.Figure:
    """Highlights the cadences removed by sigma clipping.

    Args:
        curve: A curve carrying ``meta["clipped_time"]`` and
            ``meta["clipped_flux"]`` from :class:`SigmaClipper`.

    Returns:
        The created figure.
    """
    apply_publication_style()
    figure, axis = plt.subplots()
    axis.scatter(curve.time, curve.flux, label="kept", **_RAW_STYLE)
    clipped_time = curve.meta.get("clipped_time")
    clipped_flux = curve.meta.get("clipped_flux")
    if isinstance(clipped_time, np.ndarray) and clipped_time.size:
        axis.scatter(
            clipped_time,
            clipped_flux,
            s=14.0,
            color="C3",
            marker="x",
            label=f"clipped ({clipped_time.size})",
        )
    axis.set_xlabel("Time [d]")
    axis.set_ylabel("Flux")
    axis.set_title(f"{curve.target_id} — outlier rejection")
    axis.legend(loc="best")
    return figure


def plot_normalization_comparison(curve: LightCurve) -> plt.Figure:
    """Compares the three supported normalization schemes side by side.

    Args:
        curve: A curve with finite, non-constant flux (typically taken
            just before the normalization stage).

    Returns:
        The created three-panel figure.
    """
    apply_publication_style()
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(7.0, 7.5))
    for axis, method in zip(axes, ("minmax", "median", "zscore")):
        normalized = Normalizer(method=method).apply(curve)
        axis.scatter(normalized.time, normalized.flux, **_RAW_STYLE)
        axis.set_ylabel(method)
    axes[-1].set_xlabel("Time [d]")
    axes[0].set_title(f"{curve.target_id} — normalization comparison")
    return figure


def generate_preprocessing_figures(
    raw: LightCurve,
    processed: LightCurve,
    directory: Path | str,
) -> list[Path]:
    """Exports every applicable diagnostic figure for one target.

    Figures are selected from the metadata actually present on the
    processed curve, so partial pipelines produce partial figure sets.

    Args:
        raw: The curve before preprocessing.
        processed: The pipeline output.
        directory: Output directory for figure files.

    Returns:
        Paths of all written figure files (pdf and png per figure).
    """
    directory = Path(directory)
    slug = raw.target_id.replace(" ", "_").replace("/", "-").lower()
    written: list[Path] = []

    figures: list[tuple[str, plt.Figure]] = [
        (f"{slug}_raw", plot_raw_light_curve(raw))
    ]
    if isinstance(processed.meta.get("trend"), np.ndarray):
        figures.append(
            (f"{slug}_detrended", plot_detrended_light_curve(raw, processed))
        )
    if isinstance(processed.meta.get("clipped_time"), np.ndarray):
        figures.append((f"{slug}_clipped", plot_clipped_points(processed)))
    figures.append(
        (
            f"{slug}_normalization",
            plot_normalization_comparison(_denormalized(processed)),
        )
    )

    for name, figure in figures:
        written.extend(save_figure(figure, directory, name))
        plt.close(figure)
    logger.info(
        "Target %s: wrote %d preprocessing figure file(s) to %s.",
        raw.target_id,
        len(written),
        directory,
    )
    return written
