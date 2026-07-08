"""Portable process and system resource metrics (Windows, macOS, Linux)."""

from __future__ import annotations

import os
import sys
from typing import Any

__all__ = [
    "process_rss_bytes",
    "process_cpu_seconds",
    "process_num_threads",
    "system_memory_bytes",
    "process_stats",
]


def _psutil_process() -> Any:
    import psutil

    return psutil.Process(os.getpid())


def process_rss_bytes() -> int | None:
    """Return current process resident set size in bytes."""
    try:
        return int(_psutil_process().memory_info().rss)
    except Exception:
        return None


def process_cpu_seconds() -> float | None:
    """Return accumulated user CPU time for the current process in seconds."""
    try:
        return float(_psutil_process().cpu_times().user)
    except Exception:
        return None


def process_num_threads() -> int | None:
    """Return the active thread count for the current process."""
    try:
        return int(_psutil_process().num_threads())
    except Exception:
        return None


def system_memory_bytes() -> int | None:
    """Return total physical system memory in bytes."""
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def process_stats() -> dict[str, Any]:
    """Collect a portable snapshot of process statistics for benchmarking."""
    rss = process_rss_bytes()
    cpu = process_cpu_seconds()
    threads = process_num_threads()
    return {
        "rss_bytes": rss,
        "rss_mb": round(rss / (1024**2), 2) if rss is not None else None,
        "cpu_user_seconds": cpu,
        "num_threads": threads,
        "platform": sys.platform,
    }
