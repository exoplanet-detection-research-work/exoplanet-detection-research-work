"""Filesystem and serialization helpers.

These helpers centralize directory creation, JSON round-tripping, and
file integrity checks so no other module reimplements them.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

__all__ = ["ensure_dir", "write_json", "read_json", "sha256_of_file"]

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1 << 20


def ensure_dir(path: Path | str) -> Path:
    """Creates a directory (and parents) if it does not exist.

    Args:
        path: Directory path.

    Returns:
        The directory path as a :class:`~pathlib.Path`.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(data: Any, path: Path | str, *, indent: int = 2) -> Path:
    """Serializes an object to a JSON file.

    Parent directories are created as needed. Non-JSON-native objects
    are converted with ``str`` so paths and numpy scalars serialize
    without boilerplate at call sites.

    Args:
        data: JSON-serializable object.
        path: Destination file path.
        indent: Indentation width for pretty-printing.

    Returns:
        The written file path.
    """
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, default=str, sort_keys=True)
    logger.debug("Wrote JSON: %s", path)
    return path


def read_json(path: Path | str) -> Any:
    """Reads a JSON file.

    Args:
        path: Source file path.

    Returns:
        The deserialized object.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_of_file(path: Path | str) -> str:
    """Computes the SHA-256 digest of a file, streaming in chunks.

    Used to verify integrity of downloaded datasets against published
    checksums.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
