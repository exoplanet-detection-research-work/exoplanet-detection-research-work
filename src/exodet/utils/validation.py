"""Lightweight validation helpers for numeric arrays.

Used at data-ingestion boundaries; internal code trusts its inputs.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from exodet.exceptions import DataError

__all__ = ["require_finite", "require_same_length"]


def require_finite(array: npt.NDArray[np.floating], name: str) -> None:
    """Validates that an array contains no NaN or infinite values.

    Args:
        array: Array to check.
        name: Name used in the error message.

    Raises:
        DataError: If any element is NaN or infinite.
    """
    if not np.all(np.isfinite(array)):
        bad = int(np.count_nonzero(~np.isfinite(array)))
        raise DataError(f"Array '{name}' contains {bad} non-finite value(s).")


def require_same_length(
    first: npt.NDArray[np.floating],
    second: npt.NDArray[np.floating],
    names: tuple[str, str] = ("first", "second"),
) -> None:
    """Validates that two arrays have equal length.

    Args:
        first: First array.
        second: Second array.
        names: Names used in the error message.

    Raises:
        DataError: If lengths differ.
    """
    if len(first) != len(second):
        raise DataError(
            f"Arrays '{names[0]}' ({len(first)}) and '{names[1]}' "
            f"({len(second)}) must have the same length."
        )
