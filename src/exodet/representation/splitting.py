"""Dataset splitting (Module 8).

Four strategies, all seeded and deterministic:

* ``star`` — groups samples by host star; a star's candidates land in
  exactly one split. This is the scientifically safe default: samples
  of one star share systematics, so splitting them across train/test
  leaks information.
* ``candidate`` — plain random split over samples. Allowed only when
  explicitly requested via ``allow_star_leakage=True``, otherwise the
  splitter refuses configurations that put one star on both sides.
* ``stratified`` — candidate-level split preserving label proportions
  in every split (same leakage guard).
* ``grouped`` — like ``star`` but grouping on an arbitrary metadata
  key (e.g. sector, sky region).

Every splitter returns a :class:`DatasetSplits` and every result is
checked by :func:`assert_no_group_leakage` before being returned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from exodet.exceptions import DataError, PipelineError
from exodet.registry import Registry
from exodet.representation.containers import DatasetSample

__all__ = [
    "SPLITTERS",
    "DatasetSplits",
    "StarLevelSplitter",
    "CandidateLevelSplitter",
    "StratifiedSplitter",
    "GroupedSplitter",
    "assert_no_group_leakage",
]

logger = logging.getLogger(__name__)

SPLITTERS: Registry[object] = Registry("dataset splitter")


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    """The three sample partitions of a dataset.

    Attributes:
        train: Training samples.
        validation: Validation samples.
        test: Test samples.
        meta: Split diagnostics (strategy, seed, group counts).
    """

    train: list[DatasetSample]
    validation: list[DatasetSample]
    test: list[DatasetSample]
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.train) + len(self.validation) + len(self.test)


def assert_no_group_leakage(
    splits: DatasetSplits, key: Callable[[DatasetSample], str]
) -> None:
    """Verifies that no group crosses split boundaries.

    Args:
        splits: The partitions to verify.
        key: Group key function (e.g. ``lambda s: s.target_id``).

    Raises:
        DataError: If any group appears in more than one split.
    """
    groups = [
        {key(sample) for sample in part}
        for part in (splits.train, splits.validation, splits.test)
    ]
    for i in range(3):
        for j in range(i + 1, 3):
            shared = groups[i] & groups[j]
            if shared:
                raise DataError(
                    f"Group leakage between splits: {sorted(shared)[:5]} "
                    f"appear(s) in multiple partitions."
                )


def _validate_fractions(validation_fraction: float, test_fraction: float) -> None:
    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise PipelineError("Split fractions must lie in [0, 1).")
    if validation_fraction + test_fraction >= 1:
        raise PipelineError(
            "validation_fraction + test_fraction must be < 1 "
            f"(got {validation_fraction} + {test_fraction})."
        )


def _grouped_split(
    samples: list[DatasetSample],
    key: Callable[[DatasetSample], str],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    strategy: str,
) -> DatasetSplits:
    """Assigns whole groups to splits, matching target fractions.

    Groups are shuffled deterministically and assigned greedily to the
    partition whose sample deficit (relative to its target size) is
    currently largest, which tracks the requested fractions closely
    even with very uneven group sizes.
    """
    if not samples:
        raise DataError("Cannot split an empty dataset.")
    groups: dict[str, list[DatasetSample]] = {}
    for sample in samples:
        groups.setdefault(key(sample), []).append(sample)

    names = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)

    total = len(samples)
    targets = {
        "test": test_fraction * total,
        "validation": validation_fraction * total,
        "train": (1.0 - validation_fraction - test_fraction) * total,
    }
    filled = {name: 0.0 for name in targets}
    assignment: dict[str, str] = {}
    for group_name in names:
        deficits = {
            part: (targets[part] - filled[part]) / max(targets[part], 1e-9)
            for part in targets
            if targets[part] > 0
        }
        part = max(deficits, key=deficits.get) if deficits else "train"
        assignment[group_name] = part
        filled[part] += len(groups[group_name])

    parts: dict[str, list[DatasetSample]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for group_name, part in assignment.items():
        parts[part].extend(groups[group_name])

    splits = DatasetSplits(
        train=parts["train"],
        validation=parts["validation"],
        test=parts["test"],
        meta={
            "strategy": strategy,
            "seed": seed,
            "n_groups": len(groups),
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
        },
    )
    assert_no_group_leakage(splits, key)
    logger.info(
        "%s split: %d/%d/%d samples (train/val/test) over %d group(s).",
        strategy,
        len(splits.train),
        len(splits.validation),
        len(splits.test),
        len(groups),
    )
    return splits


@SPLITTERS.register("star")
class StarLevelSplitter:
    """Splits by host star; no star ever crosses partitions.

    Attributes:
        validation_fraction: Target validation sample fraction.
        test_fraction: Target test sample fraction.
        seed: Shuffle seed.
    """

    def __init__(
        self,
        validation_fraction: float = 0.15,
        test_fraction: float = 0.15,
        seed: int = 42,
    ) -> None:
        """Initializes the splitter.

        Args:
            validation_fraction: In ``[0, 1)``.
            test_fraction: In ``[0, 1)``; sum with validation < 1.
            seed: Shuffle seed.

        Raises:
            PipelineError: If fractions are inconsistent.
        """
        _validate_fractions(validation_fraction, test_fraction)
        self.validation_fraction = float(validation_fraction)
        self.test_fraction = float(test_fraction)
        self.seed = int(seed)

    def split(self, samples: list[DatasetSample]) -> DatasetSplits:
        """Partitions the samples by star.

        Args:
            samples: The full dataset.

        Returns:
            Leakage-free splits.
        """
        return _grouped_split(
            samples,
            key=lambda s: s.target_id,
            validation_fraction=self.validation_fraction,
            test_fraction=self.test_fraction,
            seed=self.seed,
            strategy="star",
        )


@SPLITTERS.register("grouped")
class GroupedSplitter:
    """Splits by an arbitrary metadata key without group leakage.

    Attributes:
        group_key: Metadata key read from ``sample.meta`` (falls back
            to ``target_id`` when missing).
        validation_fraction: Target validation sample fraction.
        test_fraction: Target test sample fraction.
        seed: Shuffle seed.
    """

    def __init__(
        self,
        group_key: str = "group",
        validation_fraction: float = 0.15,
        test_fraction: float = 0.15,
        seed: int = 42,
    ) -> None:
        """Initializes the splitter.

        Args:
            group_key: Metadata key defining the groups.
            validation_fraction: In ``[0, 1)``.
            test_fraction: In ``[0, 1)``; sum with validation < 1.
            seed: Shuffle seed.

        Raises:
            PipelineError: If fractions are inconsistent.
        """
        _validate_fractions(validation_fraction, test_fraction)
        self.group_key = group_key
        self.validation_fraction = float(validation_fraction)
        self.test_fraction = float(test_fraction)
        self.seed = int(seed)

    def split(self, samples: list[DatasetSample]) -> DatasetSplits:
        """Partitions the samples by the configured group key.

        Args:
            samples: The full dataset.

        Returns:
            Leakage-free splits.
        """
        return _grouped_split(
            samples,
            key=lambda s: str(s.meta.get(self.group_key, s.target_id)),
            validation_fraction=self.validation_fraction,
            test_fraction=self.test_fraction,
            seed=self.seed,
            strategy=f"grouped({self.group_key})",
        )


@SPLITTERS.register("candidate")
class CandidateLevelSplitter:
    """Random candidate-level split (explicit opt-in for leakage).

    Attributes:
        validation_fraction: Target validation sample fraction.
        test_fraction: Target test sample fraction.
        seed: Shuffle seed.
        allow_star_leakage: Must be ``True`` when the split would put
            one star's candidates in different partitions.
    """

    def __init__(
        self,
        validation_fraction: float = 0.15,
        test_fraction: float = 0.15,
        seed: int = 42,
        allow_star_leakage: bool = False,
    ) -> None:
        """Initializes the splitter.

        Args:
            validation_fraction: In ``[0, 1)``.
            test_fraction: In ``[0, 1)``; sum with validation < 1.
            seed: Shuffle seed.
            allow_star_leakage: Explicit opt-in for cross-partition
                stars.

        Raises:
            PipelineError: If fractions are inconsistent.
        """
        _validate_fractions(validation_fraction, test_fraction)
        self.validation_fraction = float(validation_fraction)
        self.test_fraction = float(test_fraction)
        self.seed = int(seed)
        self.allow_star_leakage = allow_star_leakage

    def _partition(
        self, ordered: list[DatasetSample]
    ) -> tuple[list[DatasetSample], list[DatasetSample], list[DatasetSample]]:
        n = len(ordered)
        n_test = int(round(self.test_fraction * n))
        n_val = int(round(self.validation_fraction * n))
        test = ordered[:n_test]
        validation = ordered[n_test : n_test + n_val]
        train = ordered[n_test + n_val :]
        return train, validation, test

    def split(self, samples: list[DatasetSample]) -> DatasetSplits:
        """Randomly partitions the samples.

        Args:
            samples: The full dataset.

        Returns:
            The splits.

        Raises:
            DataError: If a star crosses partitions and
                ``allow_star_leakage`` is not set.
        """
        if not samples:
            raise DataError("Cannot split an empty dataset.")
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(samples))
        ordered = [samples[i] for i in order]
        train, validation, test = self._partition(ordered)
        splits = DatasetSplits(
            train=train,
            validation=validation,
            test=test,
            meta={
                "strategy": "candidate",
                "seed": self.seed,
                "allow_star_leakage": self.allow_star_leakage,
            },
        )
        if not self.allow_star_leakage:
            try:
                assert_no_group_leakage(splits, key=lambda s: s.target_id)
            except DataError as exc:
                raise DataError(
                    f"{exc} Candidate-level splitting across stars requires "
                    "allow_star_leakage=true (explicit opt-in)."
                ) from exc
        logger.info(
            "candidate split: %d/%d/%d samples (train/val/test).",
            len(train),
            len(validation),
            len(test),
        )
        return splits


@SPLITTERS.register("stratified")
class StratifiedSplitter(CandidateLevelSplitter):
    """Candidate-level split preserving label proportions per split."""

    def split(self, samples: list[DatasetSample]) -> DatasetSplits:
        """Partitions the samples with per-label proportions preserved.

        Args:
            samples: The full dataset.

        Returns:
            The stratified splits.

        Raises:
            DataError: If a star crosses partitions and
                ``allow_star_leakage`` is not set.
        """
        if not samples:
            raise DataError("Cannot split an empty dataset.")
        rng = np.random.default_rng(self.seed)
        by_label: dict[int, list[DatasetSample]] = {}
        for sample in samples:
            by_label.setdefault(sample.label, []).append(sample)

        train: list[DatasetSample] = []
        validation: list[DatasetSample] = []
        test: list[DatasetSample] = []
        for label in sorted(by_label):
            group = by_label[label]
            order = rng.permutation(len(group))
            ordered = [group[i] for i in order]
            g_train, g_val, g_test = self._partition(ordered)
            train.extend(g_train)
            validation.extend(g_val)
            test.extend(g_test)

        splits = DatasetSplits(
            train=train,
            validation=validation,
            test=test,
            meta={
                "strategy": "stratified",
                "seed": self.seed,
                "labels": sorted(by_label),
                "allow_star_leakage": self.allow_star_leakage,
            },
        )
        if not self.allow_star_leakage:
            try:
                assert_no_group_leakage(splits, key=lambda s: s.target_id)
            except DataError as exc:
                raise DataError(
                    f"{exc} Stratified splitting across stars requires "
                    "allow_star_leakage=true (explicit opt-in)."
                ) from exc
        logger.info(
            "stratified split: %d/%d/%d samples over %d label(s).",
            len(train),
            len(validation),
            len(test),
            len(by_label),
        )
        return splits
