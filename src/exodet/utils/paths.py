"""Cross-platform path and filename helpers."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

__all__ = ["safe_filename", "link_or_copy"]

_WINDOWS_RESERVED = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *[f"COM{i}" for i in range(1, 10)],
        *[f"LPT{i}" for i in range(1, 10)],
    }
)

_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00]')


def safe_filename(name: str, *, replacement: str = "_") -> str:
    """Sanitize a string for use as a file or directory name on all platforms."""
    cleaned = _INVALID_CHARS.sub(replacement, name).strip(" .")
    if not cleaned:
        cleaned = "unnamed"
    stem = cleaned.split(".")[0].upper()
    if stem in _WINDOWS_RESERVED:
        cleaned = f"_{cleaned}"
    cleaned = cleaned.rstrip(" .")
    return cleaned or "unnamed"


def link_or_copy(src: Path | str, dst: Path | str) -> Path:
    """Create a symlink to ``src`` at ``dst``, or copy if symlinks are unavailable."""
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()
    try:
        dst_path.symlink_to(src_path)
    except OSError:
        shutil.copy2(src_path, dst_path)
    return dst_path
