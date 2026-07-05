"""Dataset registry tracking processed TIC targets."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from exodet.utils.io import ensure_dir, sha256_of_file

__all__ = ["TargetRecord", "DatasetRegistry"]


@dataclass
class TargetRecord:
    """Registry entry for one processed target."""

    tic_id: str
    target_id: str
    mission: str
    download_date: str
    sectors: tuple[str, ...] = ()
    processing_version: str = "v1"
    preprocessing_version: str = "v1"
    tce_version: str = "v1"
    phase_fold_version: str = "v1"
    dataset_checksum: str = ""
    dataset_split: str = ""
    sample_ids: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TargetRecord:
        return cls(
            tic_id=str(raw["tic_id"]),
            target_id=str(raw.get("target_id", raw["tic_id"])),
            mission=str(raw.get("mission", "TESS")),
            download_date=str(raw.get("download_date", "")),
            sectors=tuple(str(s) for s in raw.get("sectors", ())),
            processing_version=str(raw.get("processing_version", "v1")),
            preprocessing_version=str(raw.get("preprocessing_version", "v1")),
            tce_version=str(raw.get("tce_version", "v1")),
            phase_fold_version=str(raw.get("phase_fold_version", "v1")),
            dataset_checksum=str(raw.get("dataset_checksum", "")),
            dataset_split=str(raw.get("dataset_split", "")),
            sample_ids=tuple(str(s) for s in raw.get("sample_ids", ())),
            meta=dict(raw.get("meta", {})),
        )


class DatasetRegistry:
    """Persistent registry of processed targets (``dataset_registry.json``)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._records: dict[str, TargetRecord] = {}
        self._load()

    def _load(self) -> None:
        if self.path.is_file():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._records = {
                k: TargetRecord.from_dict(v)
                for k, v in dict(data.get("targets", {})).items()
            }

    def _save(self) -> None:
        ensure_dir(self.path.parent)
        payload = {
            "version": 1,
            "updated_at": datetime.now(UTC).isoformat(),
            "n_targets": len(self._records),
            "targets": {k: v.to_dict() for k, v in self._records.items()},
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def normalize_tic_id(tic: str) -> str:
        """Normalize a TIC identifier to digits-only key."""
        return tic.upper().replace("TIC", "").replace(" ", "").strip()

    @staticmethod
    def format_target_id(tic: str) -> str:
        """Format canonical target_id string."""
        digits = DatasetRegistry.normalize_tic_id(tic)
        return f"TIC {digits}"

    def contains(self, tic_id: str) -> bool:
        key = self.normalize_tic_id(tic_id)
        with self._lock:
            return key in self._records

    def get(self, tic_id: str) -> TargetRecord | None:
        key = self.normalize_tic_id(tic_id)
        with self._lock:
            return self._records.get(key)

    def register(self, record: TargetRecord) -> TargetRecord:
        key = self.normalize_tic_id(record.tic_id)
        with self._lock:
            self._records[key] = record
            self._save()
        return record

    def should_process(self, tic_id: str, *, force: bool = False) -> bool:
        if force:
            return True
        return not self.contains(tic_id)

    def update_checksum(self, tic_id: str, path: Path) -> None:
        key = self.normalize_tic_id(tic_id)
        with self._lock:
            rec = self._records.get(key)
            if rec is None:
                return
            rec.dataset_checksum = sha256_of_file(path) if path.is_file() else ""
            self._records[key] = rec
            self._save()

    def iter_records(self) -> list[TargetRecord]:
        with self._lock:
            return list(self._records.values())
