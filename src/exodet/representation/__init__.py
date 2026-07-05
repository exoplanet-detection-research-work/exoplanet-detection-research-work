"""ML representation layer: TCEs → machine-learning-ready datasets.

Importing this package registers every pluggable stage:

====================  ====================  ============================
Config section        Registry              Implementations
====================  ====================  ============================
``folding``           ``PHASE_FOLDERS``     ``standard``
``global_view``       ``VIEW_BUILDERS``     ``global``
``local_view``        ``VIEW_BUILDERS``     ``local``
``physics_features``  ``PHYSICS_EXTRACTORS``  ``standard``
``scaling``           ``FEATURE_SCALERS``   ``standard``,``robust``,``minmax``
``splitting``         ``SPLITTERS``         ``star``,``candidate``,``stratified``,``grouped``
``augmentation``      ``AUGMENTERS``        ``gaussian_noise``,``timing_jitter``,``flux_scaling``,``dropout``,``cadence_mask``
====================  ====================  ============================

Pipeline flow per candidate: phase folding (+ transit alignment) →
global/local views → physics features → :class:`DatasetSample`;
then dataset-level: split (leakage-free) → scale (train-fit only) →
augment (train only) → persist.
"""

from __future__ import annotations

from exodet.representation.augmentation import (
    AUGMENTERS,
    AugmentationPipeline,
    CadenceMaskAugmenter,
    DropoutAugmenter,
    FluxScalingAugmenter,
    GaussianNoiseAugmenter,
    TimingJitterAugmenter,
)
from exodet.representation.cache import RepresentationCache, sample_fingerprint
from exodet.representation.config import (
    RepresentationConfig,
    load_representation_config,
)
from exodet.representation.containers import (
    DatasetSample,
    FeatureVector,
    PhaseFoldedCurve,
    RepresentationDataset,
    View,
)
from exodet.representation.features import PHYSICS_EXTRACTORS, PhysicsFeatureExtractor
from exodet.representation.folding import PHASE_FOLDERS, PhaseFolder, fold_phase
from exodet.representation.pipeline import RepresentationPipeline
from exodet.representation.runner import run_dataset_build
from exodet.representation.scaling import FEATURE_SCALERS, FeatureScaler
from exodet.representation.splitting import (
    SPLITTERS,
    CandidateLevelSplitter,
    DatasetSplits,
    GroupedSplitter,
    StarLevelSplitter,
    StratifiedSplitter,
    assert_no_group_leakage,
)
from exodet.representation.views import (
    VIEW_BUILDERS,
    GlobalViewGenerator,
    LocalViewGenerator,
    bin_folded_curve,
)

__all__ = [
    "AUGMENTERS",
    "FEATURE_SCALERS",
    "PHASE_FOLDERS",
    "PHYSICS_EXTRACTORS",
    "SPLITTERS",
    "VIEW_BUILDERS",
    "AugmentationPipeline",
    "CadenceMaskAugmenter",
    "CandidateLevelSplitter",
    "DatasetSample",
    "DatasetSplits",
    "DropoutAugmenter",
    "FeatureScaler",
    "FeatureVector",
    "FluxScalingAugmenter",
    "GaussianNoiseAugmenter",
    "GlobalViewGenerator",
    "GroupedSplitter",
    "LocalViewGenerator",
    "PhaseFoldedCurve",
    "PhaseFolder",
    "PhysicsFeatureExtractor",
    "RepresentationCache",
    "RepresentationConfig",
    "RepresentationDataset",
    "RepresentationPipeline",
    "StarLevelSplitter",
    "StratifiedSplitter",
    "TimingJitterAugmenter",
    "View",
    "assert_no_group_leakage",
    "bin_folded_curve",
    "fold_phase",
    "load_representation_config",
    "run_dataset_build",
    "sample_fingerprint",
]
