"""Typed configuration schema for exodet experiments.

The schema mirrors the structure of the YAML files in ``configs/``.
Every pipeline stage is described by a :class:`ComponentConfig` — a
registry ``name`` plus a free-form ``params`` mapping — so new
implementations never require schema changes.

Design notes:
    * All sections are frozen dataclasses: configuration is immutable
      after loading, which keeps experiments reproducible.
    * ``from_dict`` constructors validate structure early and raise
      :class:`~exodet.exceptions.ConfigurationError` with actionable
      messages instead of failing deep inside the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from exodet.constants import DEFAULT_RANDOM_SEED
from exodet.exceptions import ConfigurationError

__all__ = [
    "ComponentConfig",
    "PathsConfig",
    "LoggingConfig",
    "DataConfig",
    "PreprocessingConfig",
    "ModelConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "ExperimentConfig",
]


def _require_mapping(raw: Any, section: str) -> Mapping[str, Any]:
    """Validates that a config section is a mapping.

    Args:
        raw: Parsed YAML value for the section.
        section: Dotted section name for error messages.

    Returns:
        The value, narrowed to a mapping.

    Raises:
        ConfigurationError: If the value is not a mapping.
    """
    if not isinstance(raw, Mapping):
        raise ConfigurationError(
            f"Config section '{section}' must be a mapping, "
            f"got {type(raw).__name__}."
        )
    return raw


def _reject_unknown_keys(
    raw: Mapping[str, Any], allowed: frozenset[str], section: str
) -> None:
    """Raises if the mapping contains keys outside the schema.

    Args:
        raw: Parsed YAML mapping for the section.
        allowed: Set of recognised keys.
        section: Dotted section name for error messages.

    Raises:
        ConfigurationError: If unknown keys are present.
    """
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigurationError(
            f"Unknown keys in config section '{section}': "
            f"{sorted(unknown)}. Allowed: {sorted(allowed)}."
        )


@dataclass(frozen=True, slots=True)
class ComponentConfig:
    """Configuration of a single pluggable pipeline component.

    Attributes:
        name: Registry identifier of the implementation to build.
        params: Keyword arguments forwarded to the component constructor.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any, section: str) -> "ComponentConfig":
        """Builds a component config from a parsed YAML mapping.

        Args:
            raw: Mapping with a required ``name`` and optional ``params``.
            section: Dotted section name for error messages.

        Returns:
            The validated component configuration.

        Raises:
            ConfigurationError: If ``name`` is missing or keys are unknown.
        """
        mapping = _require_mapping(raw, section)
        _reject_unknown_keys(mapping, frozenset({"name", "params"}), section)
        name = mapping.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigurationError(
                f"Config section '{section}' requires a non-empty string 'name'."
            )
        params = mapping.get("params") or {}
        return cls(name=name, params=dict(_require_mapping(params, f"{section}.params")))


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Filesystem layout of the project.

    Attributes:
        data_dir: Root directory for datasets.
        raw_dir: Immutable, as-downloaded data.
        interim_dir: Intermediate artifacts of preprocessing.
        processed_dir: Model-ready datasets.
        output_dir: Root directory for run outputs.
        checkpoint_dir: Saved model weights and states.
        figure_dir: Generated figures.
        log_dir: Log files.
        report_dir: Evaluation reports and tables.
    """

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = Path("outputs/checkpoints")
    figure_dir: Path = Path("outputs/figures")
    log_dir: Path = Path("outputs/logs")
    report_dir: Path = Path("outputs/reports")

    _KEYS = frozenset(
        {
            "data_dir",
            "raw_dir",
            "interim_dir",
            "processed_dir",
            "output_dir",
            "checkpoint_dir",
            "figure_dir",
            "log_dir",
            "report_dir",
        }
    )

    @classmethod
    def from_dict(cls, raw: Any) -> "PathsConfig":
        """Builds a paths config from a parsed YAML mapping.

        Args:
            raw: Mapping of path names to path strings.

        Returns:
            The validated paths configuration.
        """
        mapping = _require_mapping(raw, "paths")
        _reject_unknown_keys(mapping, cls._KEYS, "paths")
        return cls(**{key: Path(value) for key, value in mapping.items()})


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Logging behaviour for a run.

    Attributes:
        level: Root log level name (e.g. ``"INFO"``, ``"DEBUG"``).
        to_file: Whether to also write logs to ``paths.log_dir``.
        fmt: Log record format string.
        datefmt: Timestamp format string.
    """

    level: str = "INFO"
    to_file: bool = True
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"

    _KEYS = frozenset({"level", "to_file", "fmt", "datefmt"})

    @classmethod
    def from_dict(cls, raw: Any) -> "LoggingConfig":
        """Builds a logging config from a parsed YAML mapping.

        Args:
            raw: Mapping of logging options.

        Returns:
            The validated logging configuration.
        """
        mapping = _require_mapping(raw, "logging")
        _reject_unknown_keys(mapping, cls._KEYS, "logging")
        return cls(**dict(mapping))


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Dataset selection and splitting strategy.

    Attributes:
        source: Pluggable data source (e.g. a MAST/Kaggle downloader).
        dataset: Pluggable dataset implementation that serves samples.
        train_fraction: Fraction of targets assigned to the train split.
        val_fraction: Fraction assigned to the validation split; the
            remainder forms the test split.
        stratify: Whether splits are stratified by class label.
    """

    source: ComponentConfig
    dataset: ComponentConfig
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    stratify: bool = True

    _KEYS = frozenset(
        {"source", "dataset", "train_fraction", "val_fraction", "stratify"}
    )

    @classmethod
    def from_dict(cls, raw: Any) -> "DataConfig":
        """Builds a data config from a parsed YAML mapping.

        Args:
            raw: Mapping with ``source``/``dataset`` component blocks
                and split options.

        Returns:
            The validated data configuration.

        Raises:
            ConfigurationError: If split fractions are out of range.
        """
        mapping = _require_mapping(raw, "data")
        _reject_unknown_keys(mapping, cls._KEYS, "data")
        config = cls(
            source=ComponentConfig.from_dict(mapping.get("source"), "data.source"),
            dataset=ComponentConfig.from_dict(mapping.get("dataset"), "data.dataset"),
            train_fraction=float(mapping.get("train_fraction", 0.7)),
            val_fraction=float(mapping.get("val_fraction", 0.15)),
            stratify=bool(mapping.get("stratify", True)),
        )
        if not 0.0 < config.train_fraction < 1.0:
            raise ConfigurationError(
                f"data.train_fraction must be in (0, 1), got {config.train_fraction}."
            )
        if not 0.0 <= config.val_fraction < 1.0:
            raise ConfigurationError(
                f"data.val_fraction must be in [0, 1), got {config.val_fraction}."
            )
        if config.train_fraction + config.val_fraction >= 1.0:
            raise ConfigurationError(
                "data.train_fraction + data.val_fraction must be < 1 to leave "
                "room for a test split."
            )
        return config


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Ordered sequence of preprocessing steps.

    Attributes:
        steps: Component configs applied in order to each light curve.
    """

    steps: tuple[ComponentConfig, ...] = ()

    @classmethod
    def from_dict(cls, raw: Any) -> "PreprocessingConfig":
        """Builds a preprocessing config from a parsed YAML mapping.

        Args:
            raw: Mapping with a ``steps`` list of component blocks.

        Returns:
            The validated preprocessing configuration.

        Raises:
            ConfigurationError: If ``steps`` is not a list.
        """
        mapping = _require_mapping(raw, "preprocessing")
        _reject_unknown_keys(mapping, frozenset({"steps"}), "preprocessing")
        raw_steps = mapping.get("steps", [])
        if not isinstance(raw_steps, list):
            raise ConfigurationError("preprocessing.steps must be a list.")
        steps = tuple(
            ComponentConfig.from_dict(step, f"preprocessing.steps[{i}]")
            for i, step in enumerate(raw_steps)
        )
        return cls(steps=steps)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Model selection.

    Attributes:
        architecture: Pluggable model implementation.
        features: Optional feature extractors feeding the model.
    """

    architecture: ComponentConfig
    features: tuple[ComponentConfig, ...] = ()

    _KEYS = frozenset({"architecture", "features"})

    @classmethod
    def from_dict(cls, raw: Any) -> "ModelConfig":
        """Builds a model config from a parsed YAML mapping.

        Args:
            raw: Mapping with an ``architecture`` block and optional
                ``features`` list.

        Returns:
            The validated model configuration.
        """
        mapping = _require_mapping(raw, "model")
        _reject_unknown_keys(mapping, cls._KEYS, "model")
        raw_features = mapping.get("features", [])
        if not isinstance(raw_features, list):
            raise ConfigurationError("model.features must be a list.")
        return cls(
            architecture=ComponentConfig.from_dict(
                mapping.get("architecture"), "model.architecture"
            ),
            features=tuple(
                ComponentConfig.from_dict(f, f"model.features[{i}]")
                for i, f in enumerate(raw_features)
            ),
        )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Training-loop hyperparameters.

    Attributes:
        trainer: Pluggable trainer implementation.
        epochs: Maximum number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Initial optimizer learning rate.
        early_stopping_patience: Epochs without validation improvement
            before stopping; ``0`` disables early stopping.
    """

    trainer: ComponentConfig
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10

    _KEYS = frozenset(
        {"trainer", "epochs", "batch_size", "learning_rate", "early_stopping_patience"}
    )

    @classmethod
    def from_dict(cls, raw: Any) -> "TrainingConfig":
        """Builds a training config from a parsed YAML mapping.

        Args:
            raw: Mapping with a ``trainer`` block and hyperparameters.

        Returns:
            The validated training configuration.

        Raises:
            ConfigurationError: If numeric hyperparameters are invalid.
        """
        mapping = _require_mapping(raw, "training")
        _reject_unknown_keys(mapping, cls._KEYS, "training")
        config = cls(
            trainer=ComponentConfig.from_dict(mapping.get("trainer"), "training.trainer"),
            epochs=int(mapping.get("epochs", 50)),
            batch_size=int(mapping.get("batch_size", 64)),
            learning_rate=float(mapping.get("learning_rate", 1e-3)),
            early_stopping_patience=int(mapping.get("early_stopping_patience", 10)),
        )
        if config.epochs <= 0:
            raise ConfigurationError(f"training.epochs must be > 0, got {config.epochs}.")
        if config.batch_size <= 0:
            raise ConfigurationError(
                f"training.batch_size must be > 0, got {config.batch_size}."
            )
        if config.learning_rate <= 0:
            raise ConfigurationError(
                f"training.learning_rate must be > 0, got {config.learning_rate}."
            )
        return config


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Evaluation protocol.

    Attributes:
        metrics: Metric components computed on the test split.
        decision_threshold: Probability threshold for the positive class.
    """

    metrics: tuple[ComponentConfig, ...] = ()
    decision_threshold: float = 0.5

    _KEYS = frozenset({"metrics", "decision_threshold"})

    @classmethod
    def from_dict(cls, raw: Any) -> "EvaluationConfig":
        """Builds an evaluation config from a parsed YAML mapping.

        Args:
            raw: Mapping with a ``metrics`` list and threshold.

        Returns:
            The validated evaluation configuration.

        Raises:
            ConfigurationError: If the threshold is outside ``(0, 1)``.
        """
        mapping = _require_mapping(raw, "evaluation")
        _reject_unknown_keys(mapping, cls._KEYS, "evaluation")
        raw_metrics = mapping.get("metrics", [])
        if not isinstance(raw_metrics, list):
            raise ConfigurationError("evaluation.metrics must be a list.")
        config = cls(
            metrics=tuple(
                ComponentConfig.from_dict(m, f"evaluation.metrics[{i}]")
                for i, m in enumerate(raw_metrics)
            ),
            decision_threshold=float(mapping.get("decision_threshold", 0.5)),
        )
        if not 0.0 < config.decision_threshold < 1.0:
            raise ConfigurationError(
                "evaluation.decision_threshold must be in (0, 1), "
                f"got {config.decision_threshold}."
            )
        return config


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Top-level configuration for a full experiment run.

    Attributes:
        experiment_name: Unique, filesystem-safe run identifier.
        seed: Global random seed for reproducibility.
        paths: Filesystem layout.
        logging: Logging behaviour.
        data: Dataset selection and splits.
        preprocessing: Ordered preprocessing steps.
        model: Model and feature selection.
        training: Training hyperparameters.
        evaluation: Evaluation protocol.
    """

    experiment_name: str
    seed: int
    paths: PathsConfig
    logging: LoggingConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    _KEYS = frozenset(
        {
            "experiment_name",
            "seed",
            "paths",
            "logging",
            "data",
            "preprocessing",
            "model",
            "training",
            "evaluation",
        }
    )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        """Builds the full experiment config from a parsed YAML mapping.

        Args:
            raw: Fully merged mapping of the entire config file.

        Returns:
            The validated experiment configuration.

        Raises:
            ConfigurationError: If required sections are missing or invalid.
        """
        mapping = _require_mapping(raw, "<root>")
        _reject_unknown_keys(mapping, cls._KEYS, "<root>")

        name = mapping.get("experiment_name")
        if not isinstance(name, str) or not name:
            raise ConfigurationError(
                "Top-level 'experiment_name' is required and must be a "
                "non-empty string."
            )

        for section in ("data", "model", "training"):
            if section not in mapping:
                raise ConfigurationError(f"Required config section '{section}' is missing.")

        return cls(
            experiment_name=name,
            seed=int(mapping.get("seed", DEFAULT_RANDOM_SEED)),
            paths=PathsConfig.from_dict(mapping.get("paths", {})),
            logging=LoggingConfig.from_dict(mapping.get("logging", {})),
            data=DataConfig.from_dict(mapping["data"]),
            preprocessing=PreprocessingConfig.from_dict(mapping.get("preprocessing", {})),
            model=ModelConfig.from_dict(mapping["model"]),
            training=TrainingConfig.from_dict(mapping["training"]),
            evaluation=EvaluationConfig.from_dict(mapping.get("evaluation", {})),
        )
