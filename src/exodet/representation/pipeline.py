"""Per-candidate representation pipeline (Module 7 core).

Chains folding → views → physics features into one immutable
:class:`DatasetSample`, with optional hash-validated caching. Feature
*scaling* is deliberately not part of the per-sample pipeline: scalers
must be fitted on the training split only (anything else leaks test
statistics), so the runner fits and applies them after splitting.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from exodet.data.base import LightCurve
from exodet.exceptions import DataError
from exodet.representation.cache import RepresentationCache, sample_fingerprint
from exodet.representation.config import RepresentationConfig
from exodet.representation.containers import DatasetSample
from exodet.representation.features import PHYSICS_EXTRACTORS
from exodet.representation.folding import PHASE_FOLDERS
from exodet.representation.views import VIEW_BUILDERS
from exodet.tce.candidate import TransitCandidate

__all__ = ["RepresentationPipeline"]

logger = logging.getLogger(__name__)


class RepresentationPipeline:
    """Builds ML-ready samples from (light curve, candidate) pairs.

    Attributes:
        folder: The phase folder.
        global_generator: The global view generator.
        local_generator: The local view generator.
        extractor: The physics feature extractor.
        cache: Optional sample cache.
        config: The originating configuration.
    """

    def __init__(
        self,
        config: RepresentationConfig,
        cache: RepresentationCache | None = None,
    ) -> None:
        """Builds every stage from the configuration via registries.

        Args:
            config: The validated representation configuration.
            cache: Optional cache instance (the runner wires this).

        Raises:
            RegistryError: If a configured component is not registered.
        """
        self.config = config
        self.folder = PHASE_FOLDERS.build(
            config.folding.name, **config.folding.params
        )
        self.global_generator = VIEW_BUILDERS.build(
            config.global_view.name, **config.global_view.params
        )
        self.local_generator = VIEW_BUILDERS.build(
            config.local_view.name, **config.local_view.params
        )
        self.extractor = PHYSICS_EXTRACTORS.build(
            config.physics_features.name, **config.physics_features.params
        )
        self.cache = cache
        self._signature: dict[str, Any] = {
            "folding": {"name": config.folding.name, **config.folding.params},
            "global_view": {
                "name": config.global_view.name,
                **config.global_view.params,
            },
            "local_view": {
                "name": config.local_view.name,
                **config.local_view.params,
            },
            "physics_features": {
                "name": config.physics_features.name,
                **config.physics_features.params,
            },
        }
        # Feature names are configuration-determined; probe lazily on
        # the first computed sample and reuse for cached ones.
        self._feature_names: tuple[str, ...] | None = None

    def _label_and_weight(
        self, light_curve: LightCurve, candidate: TransitCandidate
    ) -> tuple[int, float]:
        """Resolves the sample label and weight from metadata.

        Args:
            light_curve: The input curve (fallback label source).
            candidate: The candidate (primary label source).

        Returns:
            The (label, weight) pair.
        """
        labeling = self.config.labeling
        raw = candidate.meta.get(
            labeling.meta_key, light_curve.meta.get(labeling.meta_key)
        )
        label = int(raw) if raw is not None else labeling.default_label
        weight = float(labeling.label_weights.get(label, 1.0))
        return label, weight

    def build_sample(
        self, light_curve: LightCurve, candidate: TransitCandidate
    ) -> DatasetSample:
        """Builds one dataset sample, using the cache when possible.

        Args:
            light_curve: The preprocessed light curve of the target.
            candidate: The transit candidate to represent.

        Returns:
            The immutable ML-ready sample.

        Raises:
            DataError: If a view is malformed (propagated so the runner
                can record the rejection).
            PipelineError: If folding fails.
        """
        label, weight = self._label_and_weight(light_curve, candidate)
        sample_id = f"{candidate.candidate_id}_{self.config.dataset_version}"

        fingerprint = None
        if self.cache is not None:
            fingerprint = sample_fingerprint(
                light_curve, candidate, self._signature, self.config.dataset_version
            )
            cached = self.cache.get(fingerprint)
            if cached is not None:
                names = tuple(str(n) for n in np.asarray(cached["feature_names"]))
                if self._feature_names is None:
                    self._feature_names = names
                return DatasetSample(
                    sample_id=sample_id,
                    target_id=candidate.target_id,
                    candidate=candidate,
                    global_view=np.asarray(cached["global_view"]),
                    local_view=np.asarray(cached["local_view"]),
                    feature_names=names,
                    features=np.asarray(cached["features"]),
                    label=label,
                    weight=weight,
                    dataset_version=self.config.dataset_version,
                    meta={"cache_hit": True},
                    history=(*candidate.history, "representation:cache_hit"),
                )

        folded = self.folder.fold(light_curve, candidate)
        global_view = self.global_generator.generate(folded)
        local_view = self.local_generator.generate(folded)
        features = self.extractor.extract(candidate, folded, global_view, local_view)
        if self._feature_names is None:
            self._feature_names = features.names
        elif features.names != self._feature_names:
            raise DataError(
                "Inconsistent feature names across samples; the extractor "
                "configuration must be identical for the whole dataset."
            )

        local_window = float(
            local_view.bin_centers[-1] - local_view.bin_centers[0]
        )
        sample = DatasetSample(
            sample_id=sample_id,
            target_id=candidate.target_id,
            candidate=candidate,
            global_view=global_view.values,
            local_view=local_view.values,
            feature_names=features.names,
            features=features.values,
            label=label,
            weight=weight,
            dataset_version=self.config.dataset_version,
            meta={
                "cache_hit": False,
                "epoch_correction_days": folded.epoch_correction_days,
                "global_empty_fraction": global_view.empty_fraction,
                "local_empty_fraction": local_view.empty_fraction,
                "local_window_phase": local_window,
                "n_folded_cadences": len(folded),
            },
            history=(
                *folded.history,
                f"global_view(bins={global_view.n_bins})",
                f"local_view(bins={local_view.n_bins})",
                f"physics_features(n={len(features)})",
            ),
        )
        if self.cache is not None and fingerprint is not None:
            self.cache.put(
                fingerprint,
                {
                    "global_view": sample.global_view,
                    "local_view": sample.local_view,
                    "features": sample.features,
                    "feature_names": np.array(sample.feature_names, dtype=np.str_),
                },
            )
        return sample
