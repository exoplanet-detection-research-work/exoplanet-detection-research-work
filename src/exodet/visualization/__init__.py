"""Publication-quality figure helpers."""

from __future__ import annotations

from exodet.visualization.preprocessing import (
    generate_preprocessing_figures,
    plot_clipped_points,
    plot_detrended_light_curve,
    plot_normalization_comparison,
    plot_raw_light_curve,
)
from exodet.visualization.style import FIGURE_DPI, apply_publication_style, save_figure

__all__ = [
    "FIGURE_DPI",
    "apply_publication_style",
    "generate_preprocessing_figures",
    "plot_clipped_points",
    "plot_detrended_light_curve",
    "plot_normalization_comparison",
    "plot_raw_light_curve",
    "save_figure",
]
