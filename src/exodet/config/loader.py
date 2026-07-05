"""YAML configuration loading with inheritance and overrides.

Features:
    * ``defaults`` key: a config may name one or more base YAML files
      (relative to its own location) whose contents are deep-merged
      underneath it, enabling small experiment configs that override a
      shared base.
    * Dotted-key overrides: CLI flags such as
      ``--override training.epochs=100`` are applied last.
    * Scalar coercion: override values are parsed as YAML, so ``true``,
      ``3``, and ``1e-4`` become the expected Python types.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from exodet.config.schema import ExperimentConfig
from exodet.exceptions import ConfigurationError

__all__ = ["load_yaml", "deep_merge", "apply_overrides", "load_config"]

logger = logging.getLogger(__name__)

_DEFAULTS_KEY = "defaults"


def _coerce_scalar(value: str) -> Any:
    """Parses an override value string into a Python scalar.

    Values are parsed as YAML first. PyYAML follows YAML 1.1, which
    treats exponent notation without a decimal point (e.g. ``1e-4``) as
    a string, so a numeric fallback is applied for CLI ergonomics.

    Args:
        value: Raw value text from an override string.

    Returns:
        The coerced scalar (bool, int, float, str, None, ...).

    Raises:
        yaml.YAMLError: If the value is not valid YAML.
    """
    parsed = yaml.safe_load(value)
    if isinstance(parsed, str):
        try:
            return float(parsed)
        except ValueError:
            return parsed
    return parsed


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Reads a YAML file into a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed mapping; an empty file yields an empty dict.

    Raises:
        ConfigurationError: If the file is missing, unparsable, or its
            top level is not a mapping.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            content = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Failed to parse YAML file {path}: {exc}") from exc
    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ConfigurationError(
            f"Top level of {path} must be a mapping, got {type(content).__name__}."
        )
    return content


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merges ``override`` on top of ``base``.

    Nested mappings are merged key by key; any other value type in
    ``override`` (including lists) replaces the base value wholesale.

    Args:
        base: The lower-priority mapping.
        override: The higher-priority mapping.

    Returns:
        A new, independent merged dictionary.
    """
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_defaults(raw: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    """Expands the ``defaults`` key by merging base configs underneath.

    Args:
        raw: Parsed config that may contain a ``defaults`` entry.
        config_dir: Directory of the config file, used to resolve
            relative base-config paths.

    Returns:
        The config with all bases merged in and ``defaults`` removed.

    Raises:
        ConfigurationError: If ``defaults`` has an invalid type.
    """
    defaults = raw.pop(_DEFAULTS_KEY, None)
    if defaults is None:
        return raw
    if isinstance(defaults, str):
        defaults = [defaults]
    if not isinstance(defaults, list):
        raise ConfigurationError(
            f"'{_DEFAULTS_KEY}' must be a string or list of strings."
        )

    merged: dict[str, Any] = {}
    for entry in defaults:
        if not isinstance(entry, str):
            raise ConfigurationError(
                f"'{_DEFAULTS_KEY}' entries must be strings, got {entry!r}."
            )
        base_path = (config_dir / entry).resolve()
        logger.debug("Merging base config: %s", base_path)
        base_raw = _resolve_defaults(load_yaml(base_path), base_path.parent)
        merged = deep_merge(merged, base_raw)
    return deep_merge(merged, raw)


def apply_overrides(
    raw: dict[str, Any], overrides: Iterable[str]
) -> dict[str, Any]:
    """Applies ``dotted.key=value`` overrides to a parsed config.

    Args:
        raw: The parsed configuration mapping.
        overrides: Strings of the form ``section.key=value``; values are
            parsed as YAML scalars.

    Returns:
        A new mapping with all overrides applied.

    Raises:
        ConfigurationError: If an override string is malformed.
    """
    result = copy.deepcopy(raw)
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep or not key:
            raise ConfigurationError(
                f"Invalid override '{override}'; expected 'dotted.key=value'."
            )
        try:
            parsed_value = _coerce_scalar(value)
        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Invalid override value in '{override}': {exc}"
            ) from exc

        node = result
        parts = key.split(".")
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ConfigurationError(
                    f"Cannot apply override '{override}': '{part}' is not a mapping."
                )
            node = child
        node[parts[-1]] = parsed_value
        logger.debug("Applied override %s=%r", key, parsed_value)
    return result


def load_config(
    path: Path | str, overrides: Iterable[str] = ()
) -> ExperimentConfig:
    """Loads, merges, overrides, and validates an experiment config.

    Args:
        path: Path to the experiment YAML file.
        overrides: Optional ``dotted.key=value`` strings applied last.

    Returns:
        A fully validated, immutable :class:`ExperimentConfig`.

    Raises:
        ConfigurationError: If any stage of loading or validation fails.
    """
    path = Path(path)
    logger.info("Loading config: %s", path)
    raw = _resolve_defaults(load_yaml(path), path.parent)
    raw = apply_overrides(raw, overrides)
    return ExperimentConfig.from_dict(raw)
