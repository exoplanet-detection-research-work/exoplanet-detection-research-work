"""Architecture hyperparameters for exoplanet neural network models."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from exodet.exceptions import ConfigurationError

__all__ = [
    "CLASS_LABELS",
    "ModelArchitectureConfig",
    "parse_model_params",
]

CLASS_LABELS: tuple[str, ...] = (
    "transit",
    "eclipsing_binary",
    "variable_star",
    "blend",
    "noise",
)


@dataclass(frozen=True, slots=True)
class ModelArchitectureConfig:
    """Fully configurable hybrid architecture hyperparameters.

    Attributes:
        branch_mode: Which encoder branches are active.
        global_bins: Global view length (default 2001).
        local_bins: Local view length (default 401).
        n_physics_features: Physics vector dimension (0 = infer at build).
        embed_dim: Shared embedding width across branches.
        hidden_dim: Transformer FFN and fusion hidden width.
        cnn_channels: Channel widths per residual CNN stage.
        cnn_kernel_sizes: Kernel sizes paired with ``cnn_channels``.
        cnn_depthwise_stages: Stage indices using depthwise-separable conv.
        cnn_dropout: Dropout inside CNN blocks.
        transformer_depth: Number of transformer encoder layers.
        transformer_heads: Number of attention heads.
        transformer_dropout: Attention and FFN dropout.
        transformer_checkpoint: Gradient checkpointing in transformer blocks.
        physics_hidden_dims: MLP hidden layer widths for physics encoder.
        physics_dropout: Dropout in physics MLP.
        fusion_strategy: ``cross_attention``, ``gated``, or ``residual``.
        fusion_heads: Cross-attention heads in fusion module.
        fusion_dropout: Dropout in fusion layers.
        num_classes: Output classes (default 5).
        classifier_dropout: Dropout before classification head.
        confidence_hidden_dim: Hidden width of confidence MLP.
        activation: ``gelu`` or ``relu``.
        compile_model: Wrap network with ``torch.compile`` when supported.
        trainer_output: ``binary_transit`` (B,) logit for BCE trainer, or
            ``multiclass`` for raw ``(B, C)`` logits.
    """

    branch_mode: str = "fusion"
    global_bins: int = 2001
    local_bins: int = 401
    n_physics_features: int = 0
    embed_dim: int = 128
    hidden_dim: int = 256
    cnn_channels: tuple[int, ...] = (32, 64, 128)
    cnn_kernel_sizes: tuple[int, ...] = (7, 5, 3)
    cnn_depthwise_stages: tuple[int, ...] = (1, 2)
    cnn_dropout: float = 0.1
    transformer_depth: int = 4
    transformer_heads: int = 4
    transformer_dropout: float = 0.1
    transformer_checkpoint: bool = False
    physics_hidden_dims: tuple[int, ...] = (128, 64)
    physics_dropout: float = 0.1
    fusion_strategy: str = "cross_attention"
    fusion_heads: int = 4
    fusion_dropout: float = 0.1
    num_classes: int = 5
    classifier_dropout: float = 0.2
    confidence_hidden_dim: int = 64
    activation: str = "gelu"
    compile_model: bool = False
    trainer_output: str = "binary_transit"

    _BRANCHES = frozenset(
        {
            "fusion",
            "cnn_transformer",
            "cnn_only",
            "cnn",
            "transformer_only",
            "transformer",
            "physics_only",
        }
    )
    _FUSION = frozenset({"cross_attention", "gated", "residual"})
    _ACTIVATIONS = frozenset({"gelu", "relu"})
    _TRAINER_OUT = frozenset({"binary_transit", "multiclass"})

    def __post_init__(self) -> None:
        if self.branch_mode not in self._BRANCHES:
            raise ConfigurationError(
                f"branch_mode must be one of {sorted(self._BRANCHES)}, "
                f"got '{self.branch_mode}'."
            )
        if self.global_bins <= 0 or self.local_bins <= 0:
            raise ConfigurationError("global_bins and local_bins must be > 0.")
        if len(self.cnn_channels) != len(self.cnn_kernel_sizes):
            raise ConfigurationError("cnn_channels and cnn_kernel_sizes must match.")
        if self.embed_dim <= 0 or self.hidden_dim <= 0:
            raise ConfigurationError("embed_dim and hidden_dim must be > 0.")
        if self.num_classes < 2:
            raise ConfigurationError("num_classes must be >= 2.")
        if self.fusion_strategy not in self._FUSION:
            raise ConfigurationError(
                f"fusion_strategy must be one of {sorted(self._FUSION)}."
            )
        if self.activation not in self._ACTIVATIONS:
            raise ConfigurationError(
                f"activation must be one of {sorted(self._ACTIVATIONS)}."
            )
        if self.trainer_output not in self._TRAINER_OUT:
            raise ConfigurationError(
                f"trainer_output must be one of {sorted(self._TRAINER_OUT)}."
            )

    @property
    def use_cnn(self) -> bool:
        """Whether the local CNN branch is active."""
        return self.branch_mode in {
            "fusion",
            "cnn_transformer",
            "cnn_only",
            "cnn",
        }

    @property
    def use_transformer(self) -> bool:
        """Whether the global transformer branch is active."""
        return self.branch_mode in {
            "fusion",
            "cnn_transformer",
            "transformer_only",
            "transformer",
        }

    @property
    def use_physics(self) -> bool:
        """Whether the physics MLP branch is active."""
        return self.branch_mode == "fusion" or self.branch_mode == "physics_only"

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> "ModelArchitectureConfig":
        """Builds config from YAML ``model.architecture.params``."""
        raw = dict(params)
        tuple_keys = (
            "cnn_channels",
            "cnn_kernel_sizes",
            "cnn_depthwise_stages",
            "physics_hidden_dims",
        )
        for key in tuple_keys:
            if key in raw and not isinstance(raw[key], tuple):
                raw[key] = tuple(raw[key])
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


def parse_model_params(
    params: Mapping[str, Any] | None,
    input_dim: int | None = None,
    branch_mode: str | None = None,
) -> ModelArchitectureConfig:
    """Parses architecture params and infers physics dimension when needed.

    Args:
        params: YAML params dict.
        input_dim: Flattened input size from the trainer (optional).
        branch_mode: Override branch mode from registry name.

    Returns:
        Resolved architecture configuration.
    """
    mapping = dict(params or {})
    if branch_mode is not None:
        mapping["branch_mode"] = branch_mode
    config = ModelArchitectureConfig.from_mapping(mapping)
    if config.n_physics_features <= 0 and input_dim is not None:
        remaining = input_dim
        if config.use_transformer:
            remaining -= config.global_bins
        if config.use_cnn:
            remaining -= config.local_bins
        if config.use_physics and remaining > 0:
            config = replace(config, n_physics_features=remaining)
    return config
