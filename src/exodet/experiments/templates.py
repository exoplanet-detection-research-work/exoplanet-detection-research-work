"""Scientific experiment templates."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from exodet.exceptions import ConfigurationError

__all__ = [
    "EXPERIMENT_TEMPLATES",
    "apply_template",
    "list_templates",
]

EXPERIMENT_TEMPLATES: dict[str, dict[str, Any]] = {
    "cnn_baseline": {
        "description": "1-D CNN baseline on global+local views.",
        "model": {
            "architecture": {
                "name": "cnn_only",
                "params": {
                    "global_bins": 128,
                    "local_bins": 64,
                    "embed_dim": 64,
                    "cnn_channels": [16, 32],
                    "num_classes": 5,
                },
            }
        },
        "training": {
            "epochs": 20,
            "batch_size": 32,
            "learning_rate": 1.0e-3,
            "trainer": {"params": {"backend": "torch", "use_views": "both"}},
        },
        "experiments": {"tags": ["baseline", "cnn"], "stage": "train"},
    },
    "transformer_baseline": {
        "description": "Transformer-only baseline.",
        "model": {
            "architecture": {
                "name": "transformer_only",
                "params": {
                    "global_bins": 128,
                    "embed_dim": 64,
                    "transformer_depth": 2,
                    "transformer_heads": 2,
                },
            }
        },
        "training": {
            "epochs": 20,
            "batch_size": 32,
            "trainer": {"params": {"backend": "torch"}},
        },
        "experiments": {"tags": ["baseline", "transformer"], "stage": "train"},
    },
    "hybrid_model": {
        "description": "Full CNN+Transformer+Physics hybrid.",
        "model": {"architecture": {"name": "fusion", "params": {"embed_dim": 128}}},
        "training": {"epochs": 50, "batch_size": 64, "trainer": {"params": {"backend": "torch"}}},
        "experiments": {"tags": ["hybrid", "fusion"], "stage": "train"},
    },
    "ablation_study": {
        "description": "Architecture ablation campaign.",
        "ablation": {"enabled": True, "backend": "sklearn"},
        "experiments": {"tags": ["ablation"], "stage": "ablation"},
    },
    "calibration_study": {
        "description": "Calibration analysis with reliability diagrams.",
        "benchmark": {
            "enabled": True,
            "calibration": {"enabled": True, "n_bins": 15},
            "models": ["logistic_regression", "xgboost"],
        },
        "experiments": {"tags": ["calibration"], "stage": "benchmark"},
    },
    "sensitivity_study": {
        "description": "Robustness perturbation campaign.",
        "sensitivity": {"enabled": True},
        "experiments": {"tags": ["sensitivity"], "stage": "sensitivity"},
    },
    "cross_mission_eval": {
        "description": "Kepler/K2/TESS transfer evaluation.",
        "benchmark": {"enabled": True, "cross_mission": {"enabled": True}},
        "experiments": {"tags": ["cross-mission"], "stage": "benchmark"},
    },
    "sklearn_baseline": {
        "description": "Fast sklearn baseline for CI and sweeps.",
        "model": {"architecture": {"name": "logistic_regression", "params": {}}},
        "training": {
            "epochs": 1,
            "trainer": {"params": {"backend": "sklearn", "use_views": "both"}},
        },
        "experiments": {"tags": ["baseline", "sklearn"], "stage": "train"},
    },
}


def list_templates() -> dict[str, str]:
    """Return template name → description."""
    return {name: str(tpl.get("description", "")) for name, tpl in EXPERIMENT_TEMPLATES.items()}


def apply_template(name: str, config: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge a named template into a config dict."""
    if name not in EXPERIMENT_TEMPLATES:
        raise ConfigurationError(
            f"Unknown experiment template '{name}'. "
            f"Available: {sorted(EXPERIMENT_TEMPLATES)}."
        )
    merged = deepcopy(EXPERIMENT_TEMPLATES[name])
    merged.pop("description", None)

    def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        out = deepcopy(base)
        for key, value in override.items():
            if key in out and isinstance(out[key], dict) and isinstance(value, dict):
                out[key] = _merge(out[key], value)
            else:
                out[key] = deepcopy(value)
        return out

    return _merge(merged, config)
