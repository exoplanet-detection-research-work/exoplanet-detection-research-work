"""Lossless NPZ serialization of light curves.

Processed light curves are persisted as ``.npz`` archives: cadence
arrays and any NumPy arrays inside ``meta`` are stored as native
compressed arrays, while scalar/nested metadata, provenance history,
and identity fields are stored as a single JSON document. No pickling
is used, so the files are portable and safe to share alongside a
publication.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from exodet.data.base import LightCurve
from exodet.exceptions import DataError
from exodet.utils.io import ensure_dir

__all__ = ["save_light_curve", "load_light_curve"]

logger = logging.getLogger(__name__)

_META_ARRAY_PREFIX = "meta_array:"
_HEADER_KEY = "header_json"


def _json_default(value: Any) -> Any:
    """Converts NumPy scalars for JSON serialization.

    Args:
        value: A non-JSON-native object.

    Returns:
        A JSON-serializable equivalent.

    Raises:
        TypeError: If the object has no JSON equivalent.
    """
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON-serializable: {type(value).__name__}")


def save_light_curve(curve: LightCurve, path: Path | str) -> Path:
    """Writes a light curve to a compressed ``.npz`` file.

    Args:
        curve: The light curve to persist.
        path: Destination path; parent directories are created.

    Returns:
        The written file path.
    """
    path = Path(path)
    ensure_dir(path.parent)

    arrays: dict[str, np.ndarray] = {"time": curve.time, "flux": curve.flux}
    if curve.flux_err is not None:
        arrays["flux_err"] = curve.flux_err

    meta_scalars: dict[str, Any] = {}
    for key, value in curve.meta.items():
        if isinstance(value, np.ndarray):
            arrays[f"{_META_ARRAY_PREFIX}{key}"] = value
        else:
            meta_scalars[key] = value

    header = {
        "target_id": curve.target_id,
        "label": curve.label,
        "mission": curve.mission,
        "history": list(curve.history),
        "meta": meta_scalars,
    }
    arrays[_HEADER_KEY] = np.array(
        json.dumps(header, default=_json_default, sort_keys=True)
    )

    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    logger.debug("Saved light curve %s to %s", curve.target_id, path)
    return path


def load_light_curve(path: Path | str) -> LightCurve:
    """Reads a light curve written by :func:`save_light_curve`.

    Args:
        path: Source ``.npz`` file.

    Returns:
        The reconstructed light curve.

    Raises:
        DataError: If the file is missing or malformed.
    """
    path = Path(path)
    if not path.is_file():
        raise DataError(f"Light-curve file not found: {path}")

    with np.load(path, allow_pickle=False) as archive:
        try:
            header = json.loads(str(archive[_HEADER_KEY]))
            time = archive["time"]
            flux = archive["flux"]
        except KeyError as exc:
            raise DataError(f"Malformed light-curve file {path}: missing {exc}") from exc
        flux_err = archive["flux_err"] if "flux_err" in archive.files else None
        meta: dict[str, Any] = dict(header["meta"])
        for key in archive.files:
            if key.startswith(_META_ARRAY_PREFIX):
                meta[key.removeprefix(_META_ARRAY_PREFIX)] = archive[key]

    return LightCurve(
        target_id=header["target_id"],
        time=time,
        flux=flux,
        flux_err=flux_err,
        label=header["label"],
        mission=header["mission"],
        meta=meta,
        history=list(header["history"]),
    )
