"""Generic component registry enabling config-driven object construction.

Every extensible component family (data sources, preprocessors, feature
extractors, models, metrics, ...) owns a :class:`Registry` instance.
Concrete implementations register themselves by name, and the
configuration system instantiates them from YAML via that name. This is
the single mechanism through which future modules plug into the
pipeline without modifying existing code.

Example:
    >>> from exodet.registry import Registry
    >>> preprocessors: Registry[object] = Registry("preprocessor")
    >>> @preprocessors.register("median_detrend")
    ... class MedianDetrender:
    ...     pass
    >>> preprocessors.get("median_detrend")
    <class '...MedianDetrender'>
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Generic, Iterator, TypeVar

from exodet.exceptions import RegistryError

__all__ = ["Registry"]

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Registry(Generic[T]):
    """A name-to-class mapping for a single component family.

    Attributes:
        kind: Human-readable name of the component family, used in
            error messages and logging (e.g. ``"model"``).
    """

    def __init__(self, kind: str) -> None:
        """Initializes an empty registry.

        Args:
            kind: Human-readable name of the component family.
        """
        self.kind = kind
        self._entries: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Returns a class decorator that registers the class under ``name``.

        Args:
            name: Unique, case-insensitive identifier used in YAML configs.

        Returns:
            A decorator that registers and returns the class unchanged.

        Raises:
            RegistryError: If ``name`` is already registered.
        """
        key = name.lower()

        def decorator(cls: type[T]) -> type[T]:
            if key in self._entries:
                raise RegistryError(
                    f"{self.kind} '{key}' is already registered "
                    f"({self._entries[key].__qualname__})."
                )
            self._entries[key] = cls
            logger.debug("Registered %s '%s' -> %s", self.kind, key, cls.__qualname__)
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        """Looks up a registered class by name.

        Args:
            name: Case-insensitive identifier of the component.

        Returns:
            The registered class.

        Raises:
            RegistryError: If ``name`` is not registered.
        """
        key = name.lower()
        try:
            return self._entries[key]
        except KeyError:
            available = ", ".join(sorted(self._entries)) or "<none>"
            raise RegistryError(
                f"Unknown {self.kind} '{name}'. Available: {available}."
            ) from None

    def build(self, name: str, /, **kwargs: Any) -> T:
        """Instantiates a registered class with keyword arguments.

        Args:
            name: Case-insensitive identifier of the component.
            **kwargs: Keyword arguments forwarded to the constructor,
                typically taken from the ``params`` block of a YAML config.

        Returns:
            A new instance of the registered class.
        """
        cls = self.get(name)
        logger.debug("Building %s '%s' with params %s", self.kind, name, kwargs)
        return cls(**kwargs)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._entries))

    def __len__(self) -> int:
        return len(self._entries)
