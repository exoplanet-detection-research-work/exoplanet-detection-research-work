"""Typed configuration for the TCE search stage.

The TCE stage has its own YAML file (see ``configs/tce_example.yaml``),
structured like the experiment config and built from the same
primitives: :class:`~exodet.config.schema.ComponentConfig` blocks for
every pluggable stage, plus the shared ``paths`` and ``logging``
sections. Base-config inheritance (``defaults``) and dotted-key CLI
overrides work exactly as for experiment configs.
"""

from __future__ import annotations

from dataclasses import dataclass
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

__all__ = ["TCESearchConfig", "load_tce_config"]

_DEFAULT_COMPONENTS: dict[str, str] = {
    "grid": "bls_auto",
    "search": "astropy_bls",
    "peaks": "prominence",
    "detection_metrics": "standard",
    "validation": "physical",
    "harmonics": "period_ratio",
    "ranking": "metric",
}


@dataclass(frozen=True, slots=True)
class TCESearchConfig:
    """Full configuration of a TCE search run.

    Attributes:
        experiment_name: Unique, filesystem-safe run identifier.
        seed: Global random seed.
        paths: Filesystem layout (input curves are read from
            ``paths.processed_dir``).
        logging: Logging behaviour.
        grid: Search-grid generator component.
        search: BLS search engine component.
        peaks: Peak detector component.
        detection_metrics: Detection metrics computer component.
        validation: Physical validator component.
        harmonics: Harmonic rejecter component.
        ranking: Candidate ranker component.
        input_pattern: Glob for processed light-curve files.
        n_figure_targets: Leading targets for which diagnostic figures
            are exported.
    """

    experiment_name: str
    seed: int
    paths: PathsConfig
    logging: LoggingConfig
    grid: ComponentConfig
    search: ComponentConfig
    peaks: ComponentConfig
    detection_metrics: ComponentConfig
    validation: ComponentConfig
    harmonics: ComponentConfig
    ranking: ComponentConfig
    input_pattern: str = "*.npz"
    n_figure_targets: int = 1

    _KEYS = frozenset(
        {
            "experiment_name",
            "seed",
            "paths",
            "logging",
            "input_pattern",
            "n_figure_targets",
            *_DEFAULT_COMPONENTS,
        }
    )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TCESearchConfig":
        """Builds and validates the TCE config from a parsed mapping.

        Every component section is optional; omitted sections fall back
        to the default implementation with default parameters.

        Args:
            raw: Fully merged mapping of the TCE YAML file.

        Returns:
            The validated configuration.

        Raises:
            ConfigurationError: If required fields are missing or any
                section is malformed.
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

        n_figure_targets = int(mapping.get("n_figure_targets", 1))
        if n_figure_targets < 0:
            raise ConfigurationError(
                f"n_figure_targets must be >= 0, got {n_figure_targets}."
            )

        return cls(
            experiment_name=name,
            seed=int(mapping.get("seed", DEFAULT_RANDOM_SEED)),
            paths=PathsConfig.from_dict(mapping.get("paths", {})),
            logging=LoggingConfig.from_dict(mapping.get("logging", {})),
            input_pattern=str(mapping.get("input_pattern", "*.npz")),
            n_figure_targets=n_figure_targets,
            **components,
        )


def load_tce_config(
    path: Path | str, overrides: Iterable[str] = ()
) -> TCESearchConfig:
    """Loads, merges, overrides, and validates a TCE config file.

    Args:
        path: Path to the TCE YAML file.
        overrides: Optional ``dotted.key=value`` strings applied last.

    Returns:
        The validated configuration.

    Raises:
        ConfigurationError: If loading or validation fails.
    """
    path = Path(path)
    raw = _resolve_defaults(load_yaml(path), path.parent)
    raw = apply_overrides(raw, overrides)
    return TCESearchConfig.from_dict(raw)
