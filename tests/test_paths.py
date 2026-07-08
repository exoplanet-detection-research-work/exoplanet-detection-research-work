"""Tests for cross-platform path helpers."""

from __future__ import annotations

from exodet.utils.paths import safe_filename


class TestSafeFilename:
    def test_replaces_invalid_characters(self) -> None:
        assert safe_filename('TIC 1/02') == "TIC 1_02"

    def test_windows_reserved_name(self) -> None:
        assert safe_filename("CON").startswith("_")

    def test_empty_becomes_unnamed(self) -> None:
        assert safe_filename("   ") == "unnamed"
