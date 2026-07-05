"""Tests for utility modules."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from exodet.config.schema import LoggingConfig
from exodet.exceptions import DataError
from exodet.utils import (
    Timer,
    ensure_dir,
    read_json,
    require_finite,
    seed_everything,
    setup_logging,
    sha256_of_file,
    write_json,
)


class TestLogging:
    def test_setup_creates_file_handler(self, tmp_path: Path) -> None:
        logger = setup_logging(
            LoggingConfig(level="DEBUG", to_file=True),
            log_dir=tmp_path,
            run_name="test_run",
        )
        logger.info("hello")
        for handler in logger.handlers:
            handler.flush()
        log_files = list(tmp_path.glob("test_run_*.log"))
        assert len(log_files) == 1
        assert "hello" in log_files[0].read_text(encoding="utf-8")

    def test_repeated_setup_does_not_duplicate_handlers(self) -> None:
        first = setup_logging(LoggingConfig(to_file=False))
        n_handlers = len(first.handlers)
        second = setup_logging(LoggingConfig(to_file=False))
        assert len(second.handlers) == n_handlers

    def test_invalid_level_raises(self) -> None:
        with pytest.raises(ValueError, match="log level"):
            setup_logging(LoggingConfig(level="LOUD"))


class TestSeeding:
    def test_reproducible_numpy(self) -> None:
        seed_everything(1234)
        first = np.random.rand(5)
        seed_everything(1234)
        second = np.random.rand(5)
        assert np.array_equal(first, second)

    def test_negative_seed_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            seed_everything(-1)


class TestIo:
    def test_json_round_trip(self, tmp_path: Path) -> None:
        payload = {"name": "kepler", "count": 3}
        path = write_json(payload, tmp_path / "nested" / "out.json")
        assert read_json(path) == payload

    def test_sha256(self, tmp_path: Path) -> None:
        path = tmp_path / "file.bin"
        path.write_bytes(b"exoplanets")
        digest = sha256_of_file(path)
        assert len(digest) == 64
        assert digest == sha256_of_file(path)  # deterministic

    def test_ensure_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b"
        assert ensure_dir(target).is_dir()


class TestTimer:
    def test_records_elapsed(self) -> None:
        with Timer("noop", log=logging.getLogger("test")) as timer:
            pass
        assert timer.elapsed >= 0.0


class TestValidation:
    def test_require_finite_raises_on_nan(self) -> None:
        with pytest.raises(DataError, match="non-finite"):
            require_finite(np.array([1.0, np.nan]), "flux")
