"""Custom exception hierarchy for the exodet package.

All exceptions raised intentionally by this package derive from
:class:`ExoDetError`, so callers can catch package-level failures with a
single ``except`` clause while letting programming errors propagate.
"""

from __future__ import annotations

__all__ = [
    "ExoDetError",
    "ConfigurationError",
    "DataError",
    "PipelineError",
    "RegistryError",
    "NotFittedError",
]


class ExoDetError(Exception):
    """Base class for all exodet-specific errors."""


class ConfigurationError(ExoDetError):
    """Raised when a configuration file is missing, malformed, or invalid."""


class DataError(ExoDetError):
    """Raised when data is missing, corrupt, or fails validation."""


class PipelineError(ExoDetError):
    """Raised when a pipeline stage fails or stages are incompatible."""


class RegistryError(ExoDetError):
    """Raised when a component name cannot be resolved in a registry."""


class NotFittedError(ExoDetError):
    """Raised when a stateful component is used before being fitted."""
