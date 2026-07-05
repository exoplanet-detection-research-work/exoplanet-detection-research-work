"""Tests for the LightCurve container."""

from __future__ import annotations

import numpy as np
import pytest

from exodet.data.base import LightCurve
from exodet.exceptions import DataError


class TestLightCurve:
    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(DataError, match="same length"):
            LightCurve(
                target_id="X",
                time=np.arange(5, dtype=np.float64),
                flux=np.arange(4, dtype=np.float64),
            )

    def test_replace_flux_preserves_provenance(self, light_curve: LightCurve) -> None:
        new = light_curve.replace_flux(
            light_curve.flux * 2.0, step_name="doubler"
        )
        assert new.history == ["doubler"]
        assert light_curve.history == []  # original untouched
        assert new.target_id == light_curve.target_id
        assert np.allclose(new.flux, light_curve.flux * 2.0)

    def test_replace_flux_chains_history(self, light_curve: LightCurve) -> None:
        curve = light_curve.replace_flux(light_curve.flux, step_name="a")
        curve = curve.replace_flux(curve.flux, step_name="b")
        assert curve.history == ["a", "b"]

    def test_len(self, light_curve: LightCurve) -> None:
        assert len(light_curve) == 500
