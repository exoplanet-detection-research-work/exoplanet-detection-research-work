"""Tests for the component registry."""

from __future__ import annotations

import pytest

from exodet.exceptions import RegistryError
from exodet.registry import Registry


class TestRegistry:
    def test_register_and_get(self) -> None:
        registry: Registry[object] = Registry("widget")

        @registry.register("foo")
        class Foo:
            pass

        assert registry.get("foo") is Foo
        assert registry.get("FOO") is Foo  # case-insensitive
        assert "foo" in registry
        assert len(registry) == 1

    def test_build_forwards_params(self) -> None:
        registry: Registry[object] = Registry("widget")

        @registry.register("bar")
        class Bar:
            def __init__(self, value: int) -> None:
                self.value = value

        instance = registry.build("bar", value=42)
        assert isinstance(instance, Bar)
        assert instance.value == 42

    def test_duplicate_registration_raises(self) -> None:
        registry: Registry[object] = Registry("widget")

        @registry.register("dup")
        class First:
            pass

        with pytest.raises(RegistryError, match="already registered"):

            @registry.register("dup")
            class Second:
                pass

    def test_unknown_name_lists_available(self) -> None:
        registry: Registry[object] = Registry("widget")

        @registry.register("known")
        class Known:
            pass

        with pytest.raises(RegistryError, match="known"):
            registry.get("unknown")
