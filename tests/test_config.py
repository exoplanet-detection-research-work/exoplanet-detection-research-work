"""Tests for the YAML configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest

from exodet.config import ExperimentConfig, load_config
from exodet.config.loader import apply_overrides, deep_merge
from exodet.exceptions import ConfigurationError


class TestLoadConfig:
    def test_minimal_config_loads(self, config_file: Path) -> None:
        config = load_config(config_file)
        assert isinstance(config, ExperimentConfig)
        assert config.experiment_name == "unit_test"
        assert config.seed == 7
        assert config.training.epochs == 3
        assert config.model.architecture.name == "dummy_model"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_missing_required_section_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("experiment_name: x\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="'data' is missing"):
            load_config(path)

    def test_unknown_key_raises(self, config_file: Path) -> None:
        raw = config_file.read_text(encoding="utf-8") + "typo_section: 1\n"
        config_file.write_text(raw, encoding="utf-8")
        with pytest.raises(ConfigurationError, match="typo_section"):
            load_config(config_file)

    def test_overrides_applied(self, config_file: Path) -> None:
        config = load_config(
            config_file, overrides=["training.epochs=99", "seed=123"]
        )
        assert config.training.epochs == 99
        assert config.seed == 123

    def test_invalid_override_raises(self, config_file: Path) -> None:
        with pytest.raises(ConfigurationError, match="expected"):
            load_config(config_file, overrides=["no_equals_sign"])

    def test_defaults_inheritance(self, tmp_path: Path, config_file: Path) -> None:
        child = tmp_path / "child.yaml"
        child.write_text(
            f"defaults:\n  - {config_file.name}\nexperiment_name: child\n",
            encoding="utf-8",
        )
        config = load_config(child)
        assert config.experiment_name == "child"
        assert config.seed == 7  # inherited from base


class TestValidation:
    def test_bad_split_fractions(self, config_file: Path) -> None:
        with pytest.raises(ConfigurationError, match="train_fraction"):
            load_config(config_file, overrides=["data.train_fraction=1.5"])

    def test_bad_threshold(self, config_file: Path) -> None:
        with pytest.raises(ConfigurationError, match="decision_threshold"):
            load_config(config_file, overrides=["evaluation.decision_threshold=2.0"])

    def test_bad_epochs(self, config_file: Path) -> None:
        with pytest.raises(ConfigurationError, match="epochs"):
            load_config(config_file, overrides=["training.epochs=0"])


class TestDeepMerge:
    def test_nested_merge(self) -> None:
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"c": 20}, "e": 4}
        merged = deep_merge(base, override)
        assert merged == {"a": {"b": 1, "c": 20}, "d": 3, "e": 4}

    def test_inputs_not_mutated(self) -> None:
        base = {"a": {"b": 1}}
        deep_merge(base, {"a": {"b": 2}})
        assert base["a"]["b"] == 1


class TestApplyOverrides:
    def test_scalar_coercion(self) -> None:
        result = apply_overrides({}, ["x.y=1e-4", "x.flag=true", "x.n=3"])
        assert result["x"]["y"] == pytest.approx(1e-4)
        assert result["x"]["flag"] is True
        assert result["x"]["n"] == 3
