"""Light-curve preprocessing: abstractions, concrete steps, and runner.

Importing this package registers every concrete preprocessor with the
:data:`~exodet.preprocessing.base.PREPROCESSORS` registry, making them
selectable from YAML via ``preprocessing.steps``:

======================  ==========================================
Registry name           Class
======================  ==========================================
``quality_filter``      :class:`~exodet.preprocessing.cleaning.QualityFlagFilter`
``nan_removal``         :class:`~exodet.preprocessing.cleaning.NaNRemover`
``sector_stitch``       :class:`~exodet.preprocessing.stitching.SectorStitcher`
``gap_detect``          :class:`~exodet.preprocessing.gaps.GapDetector`
``gap_interpolate``     :class:`~exodet.preprocessing.gaps.GapInterpolator`
``wotan_detrend``       :class:`~exodet.preprocessing.detrending.WotanDetrender`
``sigma_clip``          :class:`~exodet.preprocessing.cleaning.SigmaClipper`
``normalize``           :class:`~exodet.preprocessing.normalization.Normalizer`
``quality_metrics``     :class:`~exodet.preprocessing.metrics.QualityMetrics`
======================  ==========================================
"""

from __future__ import annotations

from exodet.preprocessing.base import (
    PREPROCESSORS,
    BasePreprocessor,
    PreprocessingPipeline,
)
from exodet.preprocessing.cleaning import (
    TESS_QUALITY_FLAGS,
    NaNRemover,
    QualityFlagFilter,
    SigmaClipper,
)
from exodet.preprocessing.detrending import WotanDetrender
from exodet.preprocessing.gaps import GapDetector, GapInterpolator
from exodet.preprocessing.metrics import QualityMetrics, estimate_cdpp
from exodet.preprocessing.normalization import Normalizer
from exodet.preprocessing.runner import run_preprocessing
from exodet.preprocessing.stitching import SectorStitcher

__all__ = [
    "PREPROCESSORS",
    "TESS_QUALITY_FLAGS",
    "BasePreprocessor",
    "GapDetector",
    "GapInterpolator",
    "NaNRemover",
    "Normalizer",
    "PreprocessingPipeline",
    "QualityFlagFilter",
    "QualityMetrics",
    "SectorStitcher",
    "SigmaClipper",
    "WotanDetrender",
    "estimate_cdpp",
    "run_preprocessing",
]
