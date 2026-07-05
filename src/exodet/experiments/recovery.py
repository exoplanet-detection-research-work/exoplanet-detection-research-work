"""Checkpoint integrity and failure recovery."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from exodet.exceptions import PipelineError
from exodet.reproducibility.collector import checksum_file

__all__ = [
    "CheckpointIntegrity",
    "verify_checkpoint",
    "recover_interrupted_experiment",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CheckpointIntegrity:
    """Result of checkpoint verification."""

    path: str
    valid: bool
    checksum: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "valid": self.valid,
            "checksum": self.checksum,
            "reason": self.reason,
        }


def verify_checkpoint(path: Path) -> CheckpointIntegrity:
    """Verify a checkpoint file exists and is readable."""
    path = Path(path)
    if not path.is_file():
        return CheckpointIntegrity(str(path), False, "", "file missing")
    digest = checksum_file(path)
    if path.suffix == ".json":
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return CheckpointIntegrity(str(path), False, digest, f"invalid json: {exc}")
    elif path.suffix == ".pt":
        try:
            import torch

            torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            return CheckpointIntegrity(str(path), False, digest, f"torch load failed: {exc}")
    return CheckpointIntegrity(str(path), True, digest)


def recover_interrupted_experiment(
    output_dir: Path,
    *,
    state_path: Path | None = None,
) -> dict[str, Any]:
    """Load partial state from an interrupted experiment directory."""
    output_dir = Path(output_dir)
    state_path = state_path or output_dir / "state.json"
    experiment_path = output_dir / "experiment.json"
    recovery: dict[str, Any] = {
        "output_dir": str(output_dir),
        "resumable": False,
        "checkpoint": None,
        "state": None,
    }
    if state_path.is_file():
        recovery["state"] = json.loads(state_path.read_text(encoding="utf-8"))
        recovery["resumable"] = True
    if experiment_path.is_file():
        recovery["experiment"] = json.loads(experiment_path.read_text(encoding="utf-8"))

    ckpt_dir = output_dir / "checkpoints"
    for candidate in ("last.pt", "best.pt", "model.json"):
        ckpt = ckpt_dir / candidate
        if ckpt.is_file():
            integrity = verify_checkpoint(ckpt)
            if integrity.valid:
                recovery["checkpoint"] = str(ckpt)
                recovery["checkpoint_checksum"] = integrity.checksum
                recovery["resumable"] = True
                break
            logger.warning("Corrupted checkpoint %s: %s", ckpt, integrity.reason)

    if not recovery["resumable"]:
        raise PipelineError(f"No resumable state found in {output_dir}")
    return recovery
