"""Tests for the command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest

from exodet.cli.main import main


class TestCli:
    def test_info_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["info"]) == 0
        out = capsys.readouterr().out
        assert "exodet" in out
        assert "Registered components" in out

    def test_validate_config_ok(
        self, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["validate-config", "-c", str(config_file)]) == 0
        assert "OK" in capsys.readouterr().out

    def test_validate_config_missing_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["validate-config", "-c", str(tmp_path / "nope.yaml")]) == 1
        assert "error" in capsys.readouterr().err

    def test_unimplemented_stage_fails_gracefully(
        self, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["download", "-c", str(config_file)]) == 1
        assert "not implemented" in capsys.readouterr().err

    def test_override_flag(
        self, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            ["validate-config", "-c", str(config_file), "-o", "training.epochs=5"]
        )
        assert code == 0
