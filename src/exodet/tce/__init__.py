"""Transit Candidate Event (TCE) generation via Box Least Squares.

Importing this package registers every pluggable TCE stage, making
them selectable from the TCE YAML configuration:

=====================  ==================  ==============================
Config section         Registry            Implementations
=====================  ==================  ==============================
``grid``               ``GRID_GENERATORS``    ``bls_auto``
``search``             ``SEARCH_ENGINES``     ``astropy_bls``
``peaks``              ``PEAK_DETECTORS``     ``prominence``
``detection_metrics``  ``METRICS_COMPUTERS``  ``standard``
``validation``         ``VALIDATORS``         ``physical``
``harmonics``          ``HARMONIC_REJECTERS`` ``period_ratio``
``ranking``            ``RANKERS``            ``metric``, ``composite``
=====================  ==================  ==============================

The pipeline flow is: grid generation -> BLS periodogram -> peak
detection -> detection metrics -> candidate construction -> physical
validation -> harmonic rejection -> ranking. Rejected candidates are
retained with their rejection reasons for later analysis.
"""

from __future__ import annotations

from exodet.tce.candidate import Periodogram, SearchGrid, TransitCandidate
from exodet.tce.config import TCESearchConfig, load_tce_config
from exodet.tce.grid import GRID_GENERATORS, BLSGridGenerator
from exodet.tce.harmonics import HARMONIC_REJECTERS, PeriodRatioHarmonicRejecter
from exodet.tce.injection import (
    InjectionRecoveryExperiment,
    inject_box_transit,
    make_noise_light_curve,
)
from exodet.tce.metrics import METRICS_COMPUTERS, StandardMetricsComputer
from exodet.tce.peaks import PEAK_DETECTORS, ProminencePeakDetector
from exodet.tce.pipeline import TCEPipeline, TCEResult
from exodet.tce.ranking import RANKERS, CompositeRanker, MetricRanker
from exodet.tce.runner import run_tce_search
from exodet.tce.search import SEARCH_ENGINES, AstropyBLSEngine
from exodet.tce.validation import VALIDATORS, PhysicalValidator

__all__ = [
    "GRID_GENERATORS",
    "HARMONIC_REJECTERS",
    "METRICS_COMPUTERS",
    "PEAK_DETECTORS",
    "RANKERS",
    "SEARCH_ENGINES",
    "VALIDATORS",
    "AstropyBLSEngine",
    "BLSGridGenerator",
    "CompositeRanker",
    "InjectionRecoveryExperiment",
    "MetricRanker",
    "Periodogram",
    "PeriodRatioHarmonicRejecter",
    "PhysicalValidator",
    "ProminencePeakDetector",
    "SearchGrid",
    "StandardMetricsComputer",
    "TCEPipeline",
    "TCEResult",
    "TCESearchConfig",
    "TransitCandidate",
    "inject_box_transit",
    "load_tce_config",
    "make_noise_light_curve",
    "run_tce_search",
]
