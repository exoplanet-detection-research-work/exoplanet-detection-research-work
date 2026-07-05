"""Reproducibility helpers.

Seeds every random number generator the project can touch. Optional
frameworks (torch, tensorflow) are seeded only if installed, so this
module never forces heavyweight imports.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import random

import numpy as np

__all__ = ["seed_everything"]

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seeds Python, NumPy, and (if installed) torch/tensorflow RNGs.

    Args:
        seed: Non-negative seed applied to all generators.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if importlib.util.find_spec("tensorflow") is not None:
        import tensorflow as tf

        tf.random.set_seed(seed)

    logger.info("Global random seed set to %d.", seed)
