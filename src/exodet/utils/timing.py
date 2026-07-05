"""Wall-clock timing helpers for profiling pipeline stages."""

from __future__ import annotations

import functools
import logging
import time
from types import TracebackType
from typing import Any, Callable, ParamSpec, TypeVar

__all__ = ["Timer", "timed"]

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


class Timer:
    """Context manager that measures and logs elapsed wall-clock time.

    Example:
        >>> with Timer("fold light curves"):
        ...     pass  # doctest: +SKIP

    Attributes:
        label: Description of the timed block, used in the log message.
        elapsed: Elapsed seconds; populated after the block exits.
    """

    def __init__(self, label: str, *, log: logging.Logger | None = None) -> None:
        """Initializes the timer.

        Args:
            label: Description of the timed block.
            log: Logger to emit to; defaults to this module's logger.
        """
        self.label = label
        self.elapsed: float = 0.0
        self._log = log or logger
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.elapsed = time.perf_counter() - self._start
        self._log.info("%s finished in %.3f s", self.label, self.elapsed)


def timed(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator that logs the execution time of a function.

    Args:
        func: The function to wrap.

    Returns:
        A wrapper with identical signature that logs elapsed time at
        DEBUG level on the wrapped function's module logger.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logging.getLogger(func.__module__).debug(
                "%s took %.3f s", func.__qualname__, elapsed
            )

    return wrapper
