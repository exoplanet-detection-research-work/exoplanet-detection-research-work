"""Hash-validated sample cache (Module 9).

Each cache entry stores one sample's arrays (global view, local view,
features) either as a compressed ``.npz`` or as raw ``.npy`` files that
load with ``mmap_mode="r"`` (zero-copy reads for large datasets).

The cache key is a SHA-256 fingerprint over everything that determines
the sample: the light-curve content (time/flux bytes), the candidate
record, the generator configuration, and the dataset version. Any
change produces a different key, so stale entries are never returned;
the fingerprint is additionally stored *inside* each entry and
re-verified on read, guarding against manual file tampering or
collisions from renamed files. Invalid entries are deleted on sight.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from exodet.data.base import LightCurve
from exodet.exceptions import PipelineError
from exodet.tce.candidate import TransitCandidate, _jsonify
from exodet.utils.io import ensure_dir

__all__ = ["RepresentationCache", "sample_fingerprint"]

logger = logging.getLogger(__name__)

_ARRAY_KEYS = ("global_view", "local_view", "features", "feature_names")


def sample_fingerprint(
    light_curve: LightCurve,
    candidate: TransitCandidate,
    config_signature: dict[str, Any],
    dataset_version: str,
) -> str:
    """Computes the SHA-256 fingerprint identifying one sample.

    Args:
        light_curve: The input light curve (content-hashed).
        candidate: The transit candidate record.
        config_signature: All generator parameters affecting output.
        dataset_version: The dataset version tag.

    Returns:
        A 64-character hexadecimal digest.
    """
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(light_curve.time).tobytes())
    digest.update(np.ascontiguousarray(light_curve.flux).tobytes())
    payload = {
        "candidate": candidate.to_dict(),
        "config": _jsonify(config_signature),
        "version": dataset_version,
    }
    digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


class RepresentationCache:
    """Filesystem cache of generated sample arrays.

    Attributes:
        directory: Cache root directory.
        compress: Store entries as compressed NPZ (otherwise raw NPY
            files that support memory mapping).
        mmap: Load uncompressed entries memory-mapped.
    """

    def __init__(
        self,
        directory: Path | str,
        compress: bool = True,
        mmap: bool = False,
    ) -> None:
        """Initializes the cache.

        Args:
            directory: Cache root; created if missing.
            compress: Use compressed NPZ storage.
            mmap: Memory-map reads (requires ``compress=False``).

        Raises:
            PipelineError: If mmap is requested with compression.
        """
        if compress and mmap:
            raise PipelineError(
                "Memory mapping requires uncompressed storage; set "
                "compress=false to enable mmap."
            )
        self.directory = ensure_dir(directory)
        self.compress = compress
        self.mmap = mmap
        self.hits = 0
        self.misses = 0

    def _entry_dir(self, fingerprint: str) -> Path:
        # Two-level fan-out keeps directories small at 10^5+ entries.
        return self.directory / fingerprint[:2] / fingerprint

    def put(
        self, fingerprint: str, arrays: dict[str, np.ndarray]
    ) -> Path:
        """Stores one sample's arrays under its fingerprint.

        Args:
            fingerprint: The sample fingerprint (cache key).
            arrays: Mapping with keys ``global_view``, ``local_view``,
                ``features``, and ``feature_names`` (unicode array).

        Returns:
            The entry directory.

        Raises:
            PipelineError: If a required array is missing.
        """
        missing = set(_ARRAY_KEYS) - set(arrays)
        if missing:
            raise PipelineError(f"Cache entry missing arrays: {sorted(missing)}.")
        entry = ensure_dir(self._entry_dir(fingerprint))
        if self.compress:
            np.savez_compressed(
                entry / "arrays.npz",
                **{key: np.asarray(arrays[key]) for key in _ARRAY_KEYS},
            )
        else:
            for key in _ARRAY_KEYS:
                np.save(entry / f"{key}.npy", np.asarray(arrays[key]))
        (entry / "fingerprint.json").write_text(
            json.dumps({"fingerprint": fingerprint}), encoding="utf-8"
        )
        return entry

    def get(self, fingerprint: str) -> dict[str, np.ndarray] | None:
        """Retrieves a sample's arrays, validating the stored hash.

        Args:
            fingerprint: The sample fingerprint.

        Returns:
            The arrays, or ``None`` on a miss. Corrupt or stale entries
            are deleted and reported as misses.
        """
        entry = self._entry_dir(fingerprint)
        marker = entry / "fingerprint.json"
        if not marker.is_file():
            self.misses += 1
            return None
        try:
            stored = json.loads(marker.read_text(encoding="utf-8"))["fingerprint"]
            if stored != fingerprint:
                raise ValueError("fingerprint mismatch")
            if self.compress:
                with np.load(entry / "arrays.npz", allow_pickle=False) as data:
                    arrays = {key: data[key].copy() for key in _ARRAY_KEYS}
            else:
                mode = "r" if self.mmap else None
                arrays = {
                    key: np.load(entry / f"{key}.npy", mmap_mode=mode)
                    for key in _ARRAY_KEYS
                }
        except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Invalidating corrupt cache entry %s (%s).", entry.name[:12], exc
            )
            shutil.rmtree(entry, ignore_errors=True)
            self.misses += 1
            return None
        self.hits += 1
        return arrays

    def clear(self) -> int:
        """Deletes every cache entry.

        Returns:
            The number of entries removed.
        """
        n = 0
        for fanout in self.directory.iterdir():
            if fanout.is_dir():
                for entry in fanout.iterdir():
                    shutil.rmtree(entry, ignore_errors=True)
                    n += 1
                shutil.rmtree(fanout, ignore_errors=True)
        logger.info("Cleared %d cache entrie(s) from %s.", n, self.directory)
        return n

    @property
    def stats(self) -> dict[str, int | float]:
        """Hit/miss counters and hit rate for this cache instance."""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
        }
