"""Diagnostic figures for the TCE stage.

All figures follow the project publication style and are exported in
both PDF and PNG by :func:`generate_tce_figures`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exodet.data.base import LightCurve
from exodet.tce.candidate import TransitCandidate
from exodet.tce.pipeline import TCEResult
from exodet.visualization.style import apply_publication_style, save_figure

__all__ = [
    "plot_periodogram",
    "plot_power_spectrum",
    "plot_transit_markers",
    "plot_candidate_summary",
    "generate_tce_figures",
]

logger = logging.getLogger(__name__)

_POINT_STYLE = {"s": 2.0, "color": "0.3", "alpha": 0.6, "linewidths": 0.0}


def _transit_windows(
    curve: LightCurve, candidate: TransitCandidate
) -> np.ndarray:
    """Computes the mid-times of every transit inside the baseline.

    Args:
        curve: The searched light curve.
        candidate: The candidate defining period and epoch.

    Returns:
        Array of mid-transit times in days.
    """
    period = candidate.period_days
    first = candidate.epoch_days + np.ceil(
        (curve.time[0] - candidate.epoch_days - 0.5 * period) / period
    ) * period
    return np.arange(first, curve.time[-1] + 0.5 * period, period)


def _draw_periodogram(
    axis: plt.Axes, result: TCEResult, mark_candidates: bool
) -> None:
    """Renders the BLS spectrum onto an axis.

    Args:
        axis: Target matplotlib axis.
        result: The TCE search result.
        mark_candidates: Whether to mark accepted candidate periods.
    """
    pgram = result.periodogram
    axis.plot(pgram.periods, pgram.power, color="0.2", linewidth=0.7)
    axis.set_xscale("log")
    axis.set_xlabel("Period [d]")
    axis.set_ylabel(f"BLS power ({pgram.objective})")
    if mark_candidates:
        for candidate in result.accepted:
            axis.axvline(
                candidate.period_days, color="C3", alpha=0.7, linewidth=1.0
            )
            axis.annotate(
                f"{candidate.candidate_id}\nSDE={candidate.sde:.1f}",
                xy=(candidate.period_days, candidate.power),
                xytext=(4, -2),
                textcoords="offset points",
                fontsize=7,
                color="C3",
            )


def plot_periodogram(result: TCEResult) -> plt.Figure:
    """Plots BLS power versus period with candidate markers.

    Args:
        result: The TCE search result.

    Returns:
        The created figure.
    """
    apply_publication_style()
    figure, axis = plt.subplots()
    _draw_periodogram(axis, result, mark_candidates=True)
    axis.set_title(f"{result.target_id} — BLS periodogram")
    return figure


def plot_power_spectrum(result: TCEResult) -> plt.Figure:
    """Plots BLS power versus frequency.

    Args:
        result: The TCE search result.

    Returns:
        The created figure.
    """
    apply_publication_style()
    figure, axis = plt.subplots()
    pgram = result.periodogram
    axis.plot(pgram.frequencies, pgram.power, color="0.2", linewidth=0.7)
    axis.set_xlabel("Frequency [1/d]")
    axis.set_ylabel(f"BLS power ({pgram.objective})")
    axis.set_title(f"{result.target_id} — power spectrum")
    return figure


def plot_transit_markers(
    curve: LightCurve, candidate: TransitCandidate
) -> plt.Figure:
    """Plots the light curve with every predicted transit marked.

    Args:
        curve: The searched light curve.
        candidate: The candidate whose ephemeris defines the markers.

    Returns:
        The created figure.
    """
    apply_publication_style()
    figure, axis = plt.subplots()
    axis.scatter(curve.time, curve.flux, **_POINT_STYLE)
    half = 0.5 * candidate.duration_days
    for center in _transit_windows(curve, candidate):
        axis.axvspan(center - half, center + half, color="C3", alpha=0.25)
    axis.set_xlabel("Time [d]")
    axis.set_ylabel("Relative flux")
    axis.set_title(
        f"{candidate.candidate_id} — P={candidate.period_days:.4f} d, "
        f"depth={candidate.depth * 1e6:.0f} ppm"
    )
    return figure


def plot_candidate_summary(
    curve: LightCurve, result: TCEResult, candidate: TransitCandidate
) -> plt.Figure:
    """Builds the three-panel summary of the top candidate.

    Panels: full periodogram, light curve with transit markers, and a
    zoom on the first observed transit window.

    Args:
        curve: The searched light curve.
        result: The TCE search result.
        candidate: The candidate to summarize.

    Returns:
        The created figure.
    """
    apply_publication_style()
    figure, (top, middle, bottom) = plt.subplots(3, 1, figsize=(7.0, 9.0))

    _draw_periodogram(top, result, mark_candidates=True)
    top.set_title(
        f"{candidate.candidate_id}: P={candidate.period_days:.4f} d, "
        f"SDE={candidate.sde:.1f}, SNR={candidate.snr:.1f}, "
        f"depth={candidate.depth * 1e6:.0f} ppm, "
        f"{candidate.n_transits} transits"
    )

    half = 0.5 * candidate.duration_days
    centers = _transit_windows(curve, candidate)
    middle.scatter(curve.time, curve.flux, **_POINT_STYLE)
    for center in centers:
        middle.axvspan(center - half, center + half, color="C3", alpha=0.25)
    middle.set_xlabel("Time [d]")
    middle.set_ylabel("Relative flux")

    # Zoom on the first transit window that actually contains data.
    window = 3.0 * candidate.duration_days
    for center in centers:
        in_window = np.abs(curve.time - center) < window
        if np.count_nonzero(in_window) >= 3:
            bottom.scatter(
                curve.time[in_window], curve.flux[in_window], **_POINT_STYLE
            )
            bottom.axvspan(center - half, center + half, color="C3", alpha=0.25)
            bottom.set_xlim(center - window, center + window)
            break
    bottom.set_xlabel("Time [d]")
    bottom.set_ylabel("Relative flux")
    return figure


def generate_tce_figures(
    curve: LightCurve, result: TCEResult, directory: Path | str
) -> list[Path]:
    """Exports every applicable TCE diagnostic figure for one target.

    Args:
        curve: The searched light curve.
        result: The TCE search result.
        directory: Output directory.

    Returns:
        Paths of all written figure files (PDF and PNG per figure).
    """
    directory = Path(directory)
    slug = result.target_id.replace(" ", "_").replace("/", "-").lower()
    figures: list[tuple[str, plt.Figure]] = [
        (f"{slug}_bls_periodogram", plot_periodogram(result)),
        (f"{slug}_bls_power_spectrum", plot_power_spectrum(result)),
    ]
    accepted = result.accepted
    if accepted:
        best = accepted[0]
        figures.append((f"{slug}_transit_markers", plot_transit_markers(curve, best)))
        figures.append(
            (f"{slug}_top_candidate", plot_candidate_summary(curve, result, best))
        )

    written: list[Path] = []
    for name, figure in figures:
        written.extend(save_figure(figure, directory, name))
        plt.close(figure)
    logger.info(
        "Target %s: wrote %d TCE figure file(s) to %s.",
        result.target_id,
        len(written),
        directory,
    )
    return written
