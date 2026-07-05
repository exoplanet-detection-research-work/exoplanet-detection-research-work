"""Automatic device selection for PyTorch training."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from exodet.exceptions import PipelineError

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["DeviceInfo", "select_device"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Resolved compute device.

    Attributes:
        device: The torch device.
        kind: ``cpu``, ``cuda``, or ``mps``.
        index: CUDA device index (``None`` on CPU/MPS).
        name: Human-readable device description.
    """

    device: "torch.device"
    kind: str
    index: int | None
    name: str


def select_device(
    preference: str = "auto",
    cuda_index: int = 0,
) -> DeviceInfo:
    """Selects the best available PyTorch device.

    Args:
        preference: ``auto``, ``cpu``, ``cuda``, or ``mps``.
        cuda_index: CUDA device index when using CUDA.

    Returns:
        Resolved device information.

    Raises:
        PipelineError: If PyTorch is not installed or the preference is
            unavailable.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise PipelineError(
            "PyTorch is required for GPU training; install with "
            "'pip install torch'."
        ) from exc

    preference = preference.lower()
    if preference == "auto":
        if torch.cuda.is_available():
            preference = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            preference = "mps"
        else:
            preference = "cpu"

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise PipelineError("CUDA requested but not available.")
        device = torch.device(f"cuda:{cuda_index}")
        name = torch.cuda.get_device_name(cuda_index)
        logger.info("Using CUDA device %d: %s", cuda_index, name)
        return DeviceInfo(device=device, kind="cuda", index=cuda_index, name=name)

    if preference == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise PipelineError("MPS (Apple Silicon) requested but not available.")
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS backend.")
        return DeviceInfo(device=device, kind="mps", index=None, name="mps")

    if preference == "cpu":
        device = torch.device("cpu")
        logger.info("Using CPU.")
        return DeviceInfo(device=device, kind="cpu", index=None, name="cpu")

    raise PipelineError(
        f"Unknown device preference '{preference}'. Use auto|cpu|cuda|mps."
    )
