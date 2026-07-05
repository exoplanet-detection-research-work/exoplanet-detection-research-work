"""Shared helpers for preprocessing steps.

Defines the per-cadence metadata convention: some ``LightCurve.meta``
entries are arrays aligned one-to-one with the cadence arrays
(``time``/``flux``). Every step that removes or inserts cadences must
keep those entries aligned, and the helpers here are the single place
where that bookkeeping happens.

Per-cadence meta keys:
    * ``"quality"``: integer TESS quality flags.
    * ``"sector"``: integer TESS sector of each cadence.
    * ``"interpolated"``: boolean mask marking synthetic cadences.
    * ``"trend"``: trend flux removed by detrending.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import numpy.typing as npt

from exodet.data.base import LightCurve
from exodet.exceptions import PipelineError

__all__ = [
    "PER_CADENCE_META_KEYS",
    "as_float_array",
    "find_gaps",
    "mask_light_curve",
    "median_cadence",
]

logger = logging.getLogger(__name__)

PER_CADENCE_META_KEYS: Final[tuple[str, ...]] = (
    "quality",
    "sector",
    "interpolated",
    "trend",
)
"""Meta entries that are arrays aligned with the cadence axis."""


def as_float_array(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Converts input to a contiguous float64 array without copying if possible.

    Args:
        values: Any array-like of numbers.

    Returns:
        A ``float64`` NumPy array view or copy.
    """
    return np.ascontiguousarray(values, dtype=np.float64)


def mask_light_curve(
    curve: LightCurve, keep: npt.NDArray[np.bool_], step_name: str
) -> LightCurve:
    """Returns a copy of a light curve with a boolean cadence mask applied.

    The mask is applied to ``time``, ``flux``, ``flux_err``, and every
    per-cadence meta array, keeping all of them aligned. The input is
    never modified and provenance is appended via ``step_name``.

    Args:
        curve: The input light curve.
        keep: Boolean array, ``True`` for cadences to keep; must have
            the same length as the curve.
        step_name: Provenance entry recorded on the output.

    Returns:
        A new light curve containing only the kept cadences.

    Raises:
        PipelineError: If the mask length does not match the curve or
            the mask would remove every cadence.
    """
    n = len(curve)
    if keep.shape != (n,):
        raise PipelineError(
            f"Cadence mask shape {keep.shape} does not match curve length {n}."
        )
    if not keep.any():
        raise PipelineError(
            f"Step '{step_name}' would remove all {n} cadences of target "
            f"'{curve.target_id}'."
        )

    if keep.all():
        result = curve.replace_flux(curve.flux, step_name=step_name)
    else:
        result = curve.replace_flux(
            curve.flux[keep],
            step_name=step_name,
            time=curve.time[keep],
            flux_err=None if curve.flux_err is None else curve.flux_err[keep],
        )
        for key in PER_CADENCE_META_KEYS:
            value = result.meta.get(key)
            if isinstance(value, np.ndarray) and value.shape == (n,):
                result.meta[key] = value[keep]
    return result


def median_cadence(time: npt.NDArray[np.float64]) -> float:
    """Computes the median sampling interval of a time series.

    Args:
        time: Monotonically increasing observation times in days.

    Returns:
        The median time step in days.

    Raises:
        PipelineError: If fewer than two samples are available.
    """
    if time.size < 2:
        raise PipelineError("At least two cadences are required to infer cadence.")
    return float(np.median(np.diff(time)))


def find_gaps(
    time: npt.NDArray[np.float64],
    *,
    factor: float = 5.0,
    min_gap_days: float = 0.0,
) -> tuple[list[dict[str, float | int]], float]:
    """Locates observational gaps in a time series.

    A gap is any interval between consecutive cadences longer than
    ``max(factor * median_cadence, min_gap_days)``.

    Args:
        time: Monotonically increasing observation times in days.
        factor: Gap threshold as a multiple of the median cadence.
        min_gap_days: Absolute lower bound on the gap threshold in days.

    Returns:
        A tuple of (gap records, threshold in days). Each record has
        keys ``start_time``, ``end_time``, ``duration_days``, and
        ``start_index`` (index of the last cadence before the gap).
    """
    cadence = median_cadence(time)
    threshold = max(factor * cadence, min_gap_days)
    deltas = np.diff(time)
    indices = np.flatnonzero(deltas > threshold)
    gaps: list[dict[str, float | int]] = [
        {
            "start_time": float(time[i]),
            "end_time": float(time[i + 1]),
            "duration_days": float(deltas[i]),
            "start_index": int(i),
        }
        for i in indices
    ]
    return gaps, threshold
