"""Checkpoint discovery for incremental training resume."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from exodet.exceptions import PipelineError

__all__ = [
    "CheckpointDiscovery",
    "discover_checkpoint",
    "list_experiment_checkpoints",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CheckpointDiscovery:
    """Result of checkpoint search."""

    path: Path
    selection: str
    experiment_name: str
    epoch: int | None
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "selection": self.selection,
            "experiment_name": self.experiment_name,
            "epoch": self.epoch,
            "metrics": self.metrics,
        }


def list_experiment_checkpoints(checkpoint_root: Path) -> list[dict[str, Any]]:
    """List experiment subdirectories containing checkpoint files."""
    root = Path(checkpoint_root)
    if not root.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        best = sub / "best.pt"
        last = sub / "last.pt"
        if not best.is_file() and not last.is_file():
            continue
        meta_path = sub / "last.json"
        mtime = max(
            (p.stat().st_mtime for p in (best, last) if p.is_file()),
            default=0.0,
        )
        meta: dict[str, Any] = {}
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        entries.append(
            {
                "experiment_name": sub.name,
                "directory": str(sub),
                "best": str(best) if best.is_file() else None,
                "last": str(last) if last.is_file() else None,
                "mtime": mtime,
                "last_metrics": meta.get("metrics", {}),
            }
        )
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    return entries


def discover_checkpoint(
    checkpoint_root: Path,
    experiment_name: str,
    *,
    selection: str = "latest",
    explicit_path: str | None = None,
) -> CheckpointDiscovery:
    """Locate a training checkpoint for resume.

    Args:
        checkpoint_root: ``paths.checkpoint_dir`` root.
        experiment_name: Experiment subdirectory name.
        selection: ``latest``, ``best``, ``last``, or explicit filename.
        explicit_path: Override path when provided.
    """
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_file():
            raise PipelineError(f"Checkpoint not found: {path}")
        return CheckpointDiscovery(
            path=path,
            selection="explicit",
            experiment_name=experiment_name,
            epoch=None,
            metrics={},
        )

    exp_dir = Path(checkpoint_root) / experiment_name
    if selection == "latest":
        all_ckpts = list_experiment_checkpoints(checkpoint_root)
        if not all_ckpts:
            raise PipelineError(f"No checkpoints under {checkpoint_root}")
        entry = all_ckpts[0]
        exp_dir = Path(entry["directory"])
        experiment_name = entry["experiment_name"]
        path = Path(entry["last"] or entry["best"])
    elif selection == "best":
        path = exp_dir / "best.pt"
    elif selection == "last":
        path = exp_dir / "last.pt"
    else:
        path = exp_dir / selection

    if not path.is_file():
        raise PipelineError(
            f"Checkpoint '{selection}' not found at {path}; train first or choose another."
        )

    epoch = None
    metrics: dict[str, float] = {}
    meta_path = exp_dir / "last.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        epoch = int(meta.get("epoch", 0)) if meta.get("epoch") is not None else None
        metrics = {str(k): float(v) for k, v in dict(meta.get("metrics", {})).items()}

    logger.info("Discovered checkpoint %s (%s)", path, selection)
    return CheckpointDiscovery(
        path=path,
        selection=selection,
        experiment_name=experiment_name,
        epoch=epoch,
        metrics=metrics,
    )
