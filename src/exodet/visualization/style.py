"""Matplotlib style management for publication-quality figures.

Every plotting module in the project calls
:func:`apply_publication_style` once and :func:`save_figure` for
output, guaranteeing consistent typography, sizing, and export formats
across all paper figures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import matplotlib
import matplotlib.pyplot as plt

from exodet.utils.io import ensure_dir

__all__ = ["FIGURE_DPI", "apply_publication_style", "save_figure"]

logger = logging.getLogger(__name__)

FIGURE_DPI: Final[int] = 300
"""Raster export resolution suitable for journal submission."""

_STYLE: Final[dict[str, object]] = {
    "figure.figsize": (7.0, 4.5),
    "figure.dpi": 110,
    "savefig.dpi": FIGURE_DPI,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
}


def apply_publication_style() -> None:
    """Applies the project-wide matplotlib rcParams.

    Safe to call repeatedly; later calls simply reapply the style.
    """
    matplotlib.rcParams.update(_STYLE)
    logger.debug("Applied publication matplotlib style.")


def save_figure(
    figure: plt.Figure,
    directory: Path | str,
    name: str,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> list[Path]:
    """Exports a figure in every requested format.

    Args:
        figure: The matplotlib figure to save.
        directory: Output directory, created if missing.
        name: Filename stem without extension.
        formats: File formats to export.

    Returns:
        Paths of all written files.
    """
    directory = ensure_dir(directory)
    paths: list[Path] = []
    for fmt in formats:
        path = directory / f"{name}.{fmt}"
        figure.savefig(path)
        paths.append(path)
        logger.info("Saved figure: %s", path)
    return paths
