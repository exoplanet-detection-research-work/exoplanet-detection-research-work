"""ExoDet: a modular pipeline for exoplanet detection from light curves.

This package provides a config-driven, extensible architecture for
downloading, preprocessing, and classifying photometric time series
(e.g. Kepler/K2/TESS light curves) to detect transiting exoplanets.

Subpackages:
    config: YAML configuration schema and loading utilities.
    utils: Logging, I/O, seeding, timing, and validation helpers.
    data: Dataset and data-source abstractions.
    preprocessing: Light-curve preprocessing pipeline abstractions.
    features: Feature-extraction abstractions.
    models: Model interface definitions.
    training: Training-loop abstractions.
    evaluation: Metric and evaluation abstractions.
    visualization: Plotting style and figure helpers.
    cli: Command-line entrypoints.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
