"""Automatic mixed precision (Module 8)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ContextManager

from exodet.exceptions import PipelineError

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["AmpSettings", "GradScalerManager"]

logger = logging.getLogger(__name__)


def _require_torch():
    import torch

    return torch


@dataclass(frozen=True, slots=True)
class AmpSettings:
    """Resolved mixed-precision configuration.

    Attributes:
        enabled: Whether AMP is active.
        dtype: Autocast dtype (``float16`` or ``bfloat16``).
        use_scaler: Whether gradient scaling is required.
    """

    enabled: bool
    dtype: "torch.dtype | None"
    use_scaler: bool

    @classmethod
    def from_mode(cls, mode: str, device_kind: str) -> "AmpSettings":
        """Builds AMP settings from a mode string and device kind.

        Args:
            mode: ``none``, ``fp16``, or ``bf16``.
            device_kind: ``cpu``, ``cuda``, or ``mps``.

        Returns:
            Resolved AMP settings with automatic CPU/MPS fallback.
        """
        mode = mode.lower()
        if mode == "none":
            return cls(enabled=False, dtype=None, use_scaler=False)

        torch = _require_torch()
        if device_kind == "cpu":
            logger.warning("AMP disabled: CPU does not support autocast training.")
            return cls(enabled=False, dtype=None, use_scaler=False)

        if mode == "fp16":
            if device_kind == "mps":
                logger.warning("FP16 on MPS is unstable; falling back to no AMP.")
                return cls(enabled=False, dtype=None, use_scaler=False)
            return cls(enabled=True, dtype=torch.float16, use_scaler=True)

        if mode == "bf16":
            if device_kind == "mps":
                logger.warning("BF16 on MPS unavailable; falling back to no AMP.")
                return cls(enabled=False, dtype=None, use_scaler=False)
            if not torch.cuda.is_bf16_supported():
                logger.warning("BF16 unsupported on this CUDA device; using FP16.")
                return cls(enabled=True, dtype=torch.float16, use_scaler=True)
            return cls(enabled=True, dtype=torch.bfloat16, use_scaler=False)

        raise PipelineError(f"Unknown AMP mode '{mode}'.")

    def autocast(self, device_kind: str) -> ContextManager[Any]:
        """Returns an autocast context manager for the active device.

        Args:
            device_kind: ``cuda`` or ``mps``.

        Returns:
            A no-op context when AMP is disabled.
        """
        torch = _require_torch()
        if not self.enabled or self.dtype is None:
            return _nullcontext()

        device_type = "cuda" if device_kind == "cuda" else "cpu"
        if device_kind == "mps":
            device_type = "cpu"
        return torch.autocast(device_type=device_type, dtype=self.dtype)


class GradScalerManager:
    """Wraps :class:`torch.cuda.amp.GradScaler` with safe fallbacks."""

    def __init__(self, settings: AmpSettings) -> None:
        """Initializes the scaler manager.

        Args:
            settings: Resolved AMP settings.
        """
        self.settings = settings
        self._scaler: Any = None
        if settings.use_scaler:
            torch = _require_torch()
            self._scaler = torch.cuda.amp.GradScaler()

    def scale(self, loss: "torch.Tensor") -> "torch.Tensor":
        """Scales loss before backward pass when required."""
        if self._scaler is not None:
            return self._scaler.scale(loss)
        return loss

    def step(self, optimizer: "torch.optim.Optimizer") -> None:
        """Unscales gradients and steps the optimizer."""
        if self._scaler is not None:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        """Returns scaler state for checkpointing."""
        if self._scaler is None:
            return {}
        return self._scaler.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restores scaler state from a checkpoint."""
        if self._scaler is not None and state:
            self._scaler.load_state_dict(state)


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None
