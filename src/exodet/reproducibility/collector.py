"""Reproducibility metadata collection."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from exodet.config.schema import ExperimentConfig
from exodet.ml.tracking import collect_environment_info
from exodet.utils.io import sha256_of_file

__all__ = [
    "ReproducibilitySnapshot",
    "collect_reproducibility_snapshot",
    "checksum_file",
    "checksum_directory",
]


@dataclass
class ReproducibilitySnapshot:
    """Complete reproducibility record for an experiment run."""

    git_commit: str | None
    python_version: str
    package_versions: dict[str, str]
    operating_system: str
    hardware: dict[str, Any]
    random_seed: int
    configuration_snapshot: dict[str, Any]
    dataset_checksums: dict[str, str] = field(default_factory=dict)
    model_checksums: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "git_commit": self.git_commit,
            "python_version": self.python_version,
            "package_versions": self.package_versions,
            "operating_system": self.operating_system,
            "hardware": self.hardware,
            "random_seed": self.random_seed,
            "configuration_snapshot": self.configuration_snapshot,
            "dataset_checksums": self.dataset_checksums,
            "model_checksums": self.model_checksums,
            "extra": self.extra,
        }


def checksum_file(path: Path) -> str:
    """SHA-256 checksum of a file."""
    if not path.is_file():
        return ""
    return sha256_of_file(path)


def checksum_directory(path: Path, pattern: str = "*") -> str:
    """Aggregate checksum over sorted files in a directory."""
    if not path.is_dir():
        return ""
    digest = hashlib.sha256()
    for file in sorted(path.glob(pattern)):
        if file.is_file():
            digest.update(file.name.encode("utf-8"))
            digest.update(sha256_of_file(file).encode("utf-8"))
    return digest.hexdigest()


def _hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {"platform": platform.platform(), "processor": platform.processor()}
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if result.stdout.strip():
            info["memory_bytes"] = int(result.stdout.strip())
    except (FileNotFoundError, ValueError, subprocess.SubprocessError):
        pass
    return info


def _config_snapshot(experiment: ExperimentConfig, config_path: Path | None) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "experiment_name": experiment.experiment_name,
        "seed": experiment.seed,
        "model": {
            "architecture": experiment.model.architecture.name,
            "params": experiment.model.architecture.params,
        },
        "training": {
            "epochs": experiment.training.epochs,
            "batch_size": experiment.training.batch_size,
            "learning_rate": experiment.training.learning_rate,
            "trainer": experiment.training.trainer.name,
            "trainer_params": experiment.training.trainer.params,
        },
    }
    if config_path and config_path.is_file():
        snap["config_path"] = str(config_path)
        snap["config_checksum"] = checksum_file(config_path)
    return snap


def collect_reproducibility_snapshot(
    experiment: ExperimentConfig,
    *,
    config_path: Path | None = None,
    stage_settings: Mapping[str, Any] | None = None,
    dataset_paths: Mapping[str, Path] | None = None,
    model_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Collect git hash, environment, seeds, and checksums."""
    env = collect_environment_info()
    dataset_checksums = {
        name: checksum_file(path) for name, path in (dataset_paths or {}).items() if path.is_file()
    }
    model_checksums = {
        name: checksum_file(path) for name, path in (model_paths or {}).items() if path.is_file()
    }
    snapshot = ReproducibilitySnapshot(
        git_commit=env.git_commit,
        python_version=sys.version,
        package_versions=dict(env.library_versions),
        operating_system=platform.platform(),
        hardware=_hardware_info(),
        random_seed=experiment.seed,
        configuration_snapshot=_config_snapshot(experiment, config_path),
        dataset_checksums=dataset_checksums,
        model_checksums=model_checksums,
        extra=dict(stage_settings or {}),
    )
    return snapshot.to_dict()


def write_reproducibility_report(
    snapshot: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    """Write reproducibility report as JSON and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "reproducibility.json"
    json_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    md_lines = [
        "# Reproducibility Report",
        "",
        f"- Git commit: `{snapshot.get('git_commit', 'unknown')}`",
        f"- Python: `{snapshot.get('python_version', '')}`",
        f"- OS: `{snapshot.get('operating_system', '')}`",
        f"- Random seed: `{snapshot.get('random_seed', '')}`",
        "",
        "## Package versions",
        "",
    ]
    for pkg, ver in sorted(snapshot.get("package_versions", {}).items()):
        md_lines.append(f"- {pkg}: {ver}")
    md_lines.extend(["", "## Dataset checksums", ""])
    for name, digest in sorted(snapshot.get("dataset_checksums", {}).items()):
        md_lines.append(f"- {name}: `{digest}`")
    md_path = output_dir / "reproducibility.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}
