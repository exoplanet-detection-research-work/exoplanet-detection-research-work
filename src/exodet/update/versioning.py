"""Versioned dataset manifests and incremental split merging."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from exodet.representation.containers import DatasetSample, RepresentationDataset
from exodet.utils.io import ensure_dir, sha256_of_file, write_json

__all__ = [
    "DatasetManifest",
    "merge_samples",
    "append_to_splits",
    "load_or_create_manifest",
]


@dataclass
class DatasetManifest:
    """Versioned manifest for representation dataset splits."""

    version: str
    experiment_name: str
    created_at: str
    updated_at: str
    n_samples: dict[str, int] = field(default_factory=dict)
    sample_ids: dict[str, list[str]] = field(default_factory=dict)
    checksums: dict[str, str] = field(default_factory=dict)
    append_log: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "experiment_name": self.experiment_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "n_samples": self.n_samples,
            "sample_ids": self.sample_ids,
            "checksums": self.checksums,
            "append_log": self.append_log,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DatasetManifest:
        return cls(
            version=str(raw.get("version", "v1")),
            experiment_name=str(raw.get("experiment_name", "")),
            created_at=str(raw.get("created_at", "")),
            updated_at=str(raw.get("updated_at", "")),
            n_samples={str(k): int(v) for k, v in dict(raw.get("n_samples", {})).items()},
            sample_ids={str(k): list(v) for k, v in dict(raw.get("sample_ids", {})).items()},
            checksums={str(k): str(v) for k, v in dict(raw.get("checksums", {})).items()},
            append_log=list(raw.get("append_log", [])),
        )

    def save(self, path: Path) -> Path:
        return write_json(self.to_dict(), path)


def load_or_create_manifest(
    path: Path,
    *,
    version: str,
    experiment_name: str,
) -> DatasetManifest:
    if path.is_file():
        return DatasetManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    now = datetime.now(UTC).isoformat()
    return DatasetManifest(
        version=version,
        experiment_name=experiment_name,
        created_at=now,
        updated_at=now,
    )


def merge_samples(
    existing: list[DatasetSample],
    new_samples: list[DatasetSample],
) -> tuple[list[DatasetSample], list[DatasetSample]]:
    """Append samples whose sample_id is not already present."""
    seen = {s.sample_id for s in existing}
    added: list[DatasetSample] = []
    merged = list(existing)
    for sample in new_samples:
        if sample.sample_id in seen:
            continue
        merged.append(sample)
        added.append(sample)
        seen.add(sample.sample_id)
    return merged, added


def append_to_splits(
    dataset_dir: Path,
    new_samples: list[DatasetSample],
    *,
    split: str,
    version: str,
    experiment_name: str,
    manifest_path: Path,
) -> dict[str, Any]:
    """Append new samples to one split without rebuilding others."""
    dataset_dir = Path(dataset_dir)
    ensure_dir(dataset_dir)
    split_path = dataset_dir / f"{split}.npz"
    existing = (
        RepresentationDataset.load(split_path)
        if split_path.is_file()
        else RepresentationDataset([], version=version)
    )
    merged, added = merge_samples(list(existing.samples), new_samples)
    if not added:
        return {"split": split, "n_added": 0, "n_total": len(existing)}

    out = RepresentationDataset(
        merged,
        version=version,
        meta={**existing.meta, "split": split, "experiment": experiment_name},
    )
    out.save(split_path)

    manifest = load_or_create_manifest(
        manifest_path, version=version, experiment_name=experiment_name
    )
    manifest.n_samples[split] = len(merged)
    manifest.sample_ids[split] = [s.sample_id for s in merged]
    manifest.checksums[split] = sha256_of_file(split_path)
    manifest.updated_at = datetime.now(UTC).isoformat()
    manifest.append_log.append(
        {
            "timestamp": manifest.updated_at,
            "split": split,
            "n_added": len(added),
            "sample_ids": [s.sample_id for s in added],
        }
    )
    manifest.save(manifest_path)
    return {
        "split": split,
        "n_added": len(added),
        "n_total": len(merged),
        "added_sample_ids": [s.sample_id for s in added],
    }
