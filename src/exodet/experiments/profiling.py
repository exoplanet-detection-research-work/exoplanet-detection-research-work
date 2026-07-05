"""Hardware and runtime profiling for experiment runs."""

from __future__ import annotations

import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from exodet.ml.tracking import collect_environment_info
from exodet.reproducibility.collector import _hardware_info

__all__ = ["HardwareProfile", "ProfileContext", "collect_hardware_profile"]


@dataclass
class HardwareProfile:
    """Hardware and software execution context."""

    cpu_count: int
    platform: str
    processor: str
    ram_bytes: int | None
    disk_free_bytes: int | None
    cuda_available: bool
    cuda_version: str | None
    cuda_device: str | None
    torch_version: str | None
    python_version: str
    git_commit: str | None
    package_versions: dict[str, str] = field(default_factory=dict)
    energy_estimate_joules: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_count": self.cpu_count,
            "platform": self.platform,
            "processor": self.processor,
            "ram_bytes": self.ram_bytes,
            "disk_free_bytes": self.disk_free_bytes,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "cuda_device": self.cuda_device,
            "torch_version": self.torch_version,
            "python_version": self.python_version,
            "git_commit": self.git_commit,
            "package_versions": self.package_versions,
            "energy_estimate_joules": self.energy_estimate_joules,
        }


def collect_hardware_profile(*, output_dir: Any = None) -> HardwareProfile:
    """Collect CPU/GPU/RAM/disk and library versions."""
    env = collect_environment_info()
    hw = _hardware_info()
    disk_free = None
    if output_dir is not None:
        try:
            usage = shutil.disk_usage(Path(output_dir))
            disk_free = int(usage.free)
        except OSError:
            pass

    cuda_available = False
    cuda_version = hw.get("cuda_version")
    cuda_device = hw.get("cuda_device")
    torch_version = env.library_versions.get("torch")
    if torch_version and torch_version != "not installed":
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            if cuda_available and not cuda_device:
                cuda_device = torch.cuda.get_device_name(0)
            if not cuda_version:
                cuda_version = torch.version.cuda
        except ImportError:
            pass

    return HardwareProfile(
        cpu_count=os.cpu_count() or 1,
        platform=platform.platform(),
        processor=platform.processor(),
        ram_bytes=hw.get("memory_bytes"),
        disk_free_bytes=disk_free,
        cuda_available=cuda_available,
        cuda_version=str(cuda_version) if cuda_version else None,
        cuda_device=str(cuda_device) if cuda_device else None,
        torch_version=torch_version if torch_version != "not installed" else None,
        python_version=sys.version.split()[0],
        git_commit=env.git_commit,
        package_versions=dict(env.library_versions),
    )


@dataclass
class ProfileContext:
    """Context manager measuring wall-clock runtime."""

    output_dir: Any = None
    tdp_watts: float = 65.0

    def __post_init__(self) -> None:
        self._start = 0.0
        self.elapsed_seconds = 0.0
        self.hardware: HardwareProfile | None = None

    def __enter__(self) -> "ProfileContext":
        self.hardware = collect_hardware_profile(output_dir=self.output_dir)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed_seconds = time.perf_counter() - self._start
        if self.hardware is not None:
            self.hardware.energy_estimate_joules = self.tdp_watts * self.elapsed_seconds
