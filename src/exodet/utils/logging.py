"""Centralized logging configuration.

Call :func:`setup_logging` exactly once at process start (the CLI does
this automatically). All modules obtain loggers the standard way::

    logger = logging.getLogger(__name__)

which places them under the ``exodet`` namespace so verbosity can be
controlled package-wide.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from exodet.config.schema import LoggingConfig
from exodet.constants import PACKAGE_NAME

__all__ = ["setup_logging"]


def setup_logging(
    config: LoggingConfig | None = None,
    log_dir: Path | None = None,
    run_name: str | None = None,
) -> logging.Logger:
    """Configures handlers and formatting for the package logger.

    Repeated calls replace existing handlers, so the function is safe
    to call from tests and notebooks.

    Args:
        config: Logging options; defaults are used when omitted.
        log_dir: Directory for the log file. A file handler is added
            only if this is provided and ``config.to_file`` is true.
        run_name: Optional run identifier included in the log filename.

    Returns:
        The configured ``exodet`` package logger.

    Raises:
        ValueError: If ``config.level`` is not a valid level name.
    """
    config = config or LoggingConfig()

    level = logging.getLevelName(config.level.upper())
    if not isinstance(level, int):
        raise ValueError(f"Unknown log level: {config.level!r}")

    package_logger = logging.getLogger(PACKAGE_NAME)
    package_logger.setLevel(level)
    package_logger.handlers.clear()
    package_logger.propagate = False

    formatter = logging.Formatter(fmt=config.fmt, datefmt=config.datefmt)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(formatter)
    package_logger.addHandler(console)

    if config.to_file and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = f"{run_name}_{timestamp}" if run_name else timestamp
        file_handler = logging.FileHandler(log_dir / f"{stem}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        package_logger.addHandler(file_handler)
        package_logger.debug("Logging to file: %s", log_dir / f"{stem}.log")

    return package_logger
