"""Typed configuration of the representation (dataset) stage.

Mirrors the TCE config style: its own YAML file
(``configs/representation_example.yaml``), built from the shared
:class:`~exodet.config.schema.ComponentConfig`, ``paths``, and
``logging`` primitives, with ``defaults`` inheritance and dotted-key
CLI overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from exodet.config.loader import _resolve_defaults, apply_overrides, load_yaml
from exodet.config.schema import (
    ComponentConfig,
    LoggingConfig,
    PathsConfig,
    _reject_unknown_keys,
    _require_mapping,
)
from exodet.constants import DEFAULT_RANDOM_SEED
from exodet.exceptions import ConfigurationError

__all__ = ["RepresentationConfig", "load_representation_config"]

_DEFAULT_COMPONENTS: dict[str, str] = {
    "folding": "standard",
    "global_view": "global",
    "local_view": "local",
    "physics_features": "standard",
    "scaling": "standard",
    "splitting": "star",
}


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """Cache behaviour of the dataset builder.

    Attributes:
        enabled: Whether caching is active.
        directory: Cache root (``null`` → ``<interim_dir>/rep_cache``).
        compress: Compressed NPZ storage (otherwise mmap-able NPY).
        mmap: Memory-map reads of uncompressed entries.
    """

    enabled: bool = True
    directory: str | None = None
    compress: bool = True
    mmap: bool = False

    _KEYS = frozenset({"enabled", "directory", "compress", "mmap"})

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CacheConfig":
        """Builds the cache config from a mapping.

        Args:
            raw: The ``cache`` section.

        Returns:
            The validated config.
        """
        mapping = _require_mapping(raw, "cache")
        _reject_unknown_keys(mapping, cls._KEYS, "cache")
        return cls(
            enabled=bool(mapping.get("enabled", True)),
            directory=mapping.get("directory"),
            compress=bool(mapping.get("compress", True)),
            mmap=bool(mapping.get("mmap", False)),
        )


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """Training-split augmentation behaviour.

    Attributes:
        enabled: Whether augmentation runs at all.
        copies: Augmented copies per training sample.
        steps: Augmenter components applied in order.
    """

    enabled: bool = False
    copies: int = 1
    steps: tuple[ComponentConfig, ...] = ()

    _KEYS = frozenset({"enabled", "copies", "steps"})

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AugmentationConfig":
        """Builds the augmentation config from a mapping.

        Args:
            raw: The ``augmentation`` section.

        Returns:
            The validated config.

        Raises:
            ConfigurationError: If ``copies`` is negative or steps are
                malformed.
        """
        mapping = _require_mapping(raw, "augmentation")
        _reject_unknown_keys(mapping, cls._KEYS, "augmentation")
        copies = int(mapping.get("copies", 1))
        if copies < 0:
            raise ConfigurationError(f"augmentation.copies must be >= 0, got {copies}.")
        raw_steps = mapping.get("steps", [])
        if not isinstance(raw_steps, list):
            raise ConfigurationError("augmentation.steps must be a list.")
        steps = tuple(
            ComponentConfig.from_dict(step, f"augmentation.steps[{index}]")
            for index, step in enumerate(raw_steps)
        )
        return cls(
            enabled=bool(mapping.get("enabled", False)),
            copies=copies,
            steps=steps,
        )


@dataclass(frozen=True, slots=True)
class LabelingConfig:
    """How sample labels and weights are assigned.

    Attributes:
        meta_key: Key looked up in ``candidate.meta`` (then the light
            curve ``meta``) for the integer label.
        default_label: Label when the key is absent (−1 = unlabeled).
        label_weights: Optional per-label sample weights.
    """

    meta_key: str = "label"
    default_label: int = -1
    label_weights: dict[int, float] = field(default_factory=dict)

    _KEYS = frozenset({"meta_key", "default_label", "label_weights"})

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LabelingConfig":
        """Builds the labeling config from a mapping.

        Args:
            raw: The ``labeling`` section.

        Returns:
            The validated config.
        """
        mapping = _require_mapping(raw, "labeling")
        _reject_unknown_keys(mapping, cls._KEYS, "labeling")
        weights_raw = _require_mapping(
            mapping.get("label_weights", {}), "labeling.label_weights"
        )
        return cls(
            meta_key=str(mapping.get("meta_key", "label")),
            default_label=int(mapping.get("default_label", -1)),
            label_weights={int(k): float(v) for k, v in weights_raw.items()},
        )


@dataclass(frozen=True, slots=True)
class RepresentationConfig:
    """Full configuration of a dataset-build run.

    Attributes:
        experiment_name: Unique, filesystem-safe run identifier.
        seed: Global random seed.
        paths: Filesystem layout (curves from ``processed_dir``, TCE
            catalog from ``report_dir``).
        logging: Logging behaviour.
        folding: Phase folder component.
        global_view: Global view generator component.
        local_view: Local view generator component.
        physics_features: Feature extractor component.
        scaling: Feature scaler component.
        splitting: Dataset splitter component.
        cache: Caching behaviour.
        augmentation: Training-split augmentation.
        labeling: Label/weight assignment.
        dataset_version: Version tag stamped on every sample.
        candidates_file: TCE catalog filename inside ``report_dir``.
        accepted_only: Build samples only for accepted candidates.
        n_figure_samples: Leading samples with diagnostic figures.
    """

    experiment_name: str
    seed: int
    paths: PathsConfig
    logging: LoggingConfig
    folding: ComponentConfig
    global_view: ComponentConfig
    local_view: ComponentConfig
    physics_features: ComponentConfig
    scaling: ComponentConfig
    splitting: ComponentConfig
    cache: CacheConfig
    augmentation: AugmentationConfig
    labeling: LabelingConfig
    dataset_version: str = "v1"
    candidates_file: str = "tce_candidates.json"
    accepted_only: bool = True
    n_figure_samples: int = 1

    _KEYS = frozenset(
        {
            "experiment_name",
            "seed",
            "paths",
            "logging",
            "cache",
            "augmentation",
            "labeling",
            "dataset_version",
            "candidates_file",
            "accepted_only",
            "n_figure_samples",
            *_DEFAULT_COMPONENTS,
        }
    )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RepresentationConfig":
        """Builds and validates the config from a parsed mapping.

        Args:
            raw: Fully merged mapping of the YAML file.

        Returns:
            The validated configuration.

        Raises:
            ConfigurationError: On missing/unknown/malformed fields.
        """
        mapping = _require_mapping(raw, "<root>")
        _reject_unknown_keys(mapping, cls._KEYS, "<root>")

        name = mapping.get("experiment_name")
        if not isinstance(name, str) or not name:
            raise ConfigurationError(
                "Top-level 'experiment_name' is required and must be a "
                "non-empty string."
            )
        components: dict[str, ComponentConfig] = {}
        for section, default_name in _DEFAULT_COMPONENTS.items():
            block = mapping.get(section, {"name": default_name})
            components[section] = ComponentConfig.from_dict(block, section)

        n_figure_samples = int(mapping.get("n_figure_samples", 1))
        if n_figure_samples < 0:
            raise ConfigurationError(
                f"n_figure_samples must be >= 0, got {n_figure_samples}."
            )

        return cls(
            experiment_name=name,
            seed=int(mapping.get("seed", DEFAULT_RANDOM_SEED)),
            paths=PathsConfig.from_dict(mapping.get("paths", {})),
            logging=LoggingConfig.from_dict(mapping.get("logging", {})),
            cache=CacheConfig.from_dict(mapping.get("cache", {})),
            augmentation=AugmentationConfig.from_dict(
                mapping.get("augmentation", {})
            ),
            labeling=LabelingConfig.from_dict(mapping.get("labeling", {})),
            dataset_version=str(mapping.get("dataset_version", "v1")),
            candidates_file=str(
                mapping.get("candidates_file", "tce_candidates.json")
            ),
            accepted_only=bool(mapping.get("accepted_only", True)),
            n_figure_samples=n_figure_samples,
            **components,
        )


def load_representation_config(
    path: Path | str, overrides: Iterable[str] = ()
) -> RepresentationConfig:
    """Loads, merges, overrides, and validates a representation config.

    Args:
        path: Path to the YAML file.
        overrides: Optional ``dotted.key=value`` strings applied last.

    Returns:
        The validated configuration.

    Raises:
        ConfigurationError: If loading or validation fails.
    """
    path = Path(path)
    raw = _resolve_defaults(load_yaml(path), path.parent)
    raw = apply_overrides(raw, overrides)
    return RepresentationConfig.from_dict(raw)
