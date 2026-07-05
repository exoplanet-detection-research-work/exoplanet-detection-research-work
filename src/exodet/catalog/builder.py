"""Searchable exoplanet candidate catalog."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from exodet.inference.config import CatalogStageConfig
from exodet.inference.containers import ScientificInferenceBatch
from exodet.utils.io import ensure_dir, write_json

__all__ = ["CatalogEntry", "CatalogBuilder"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One catalog row for a detected candidate."""

    tic_id: str
    target_id: str
    sample_id: str
    candidate_id: str
    classification: str
    confidence: float
    probability: float
    transit: dict[str, Any] = field(default_factory=dict)
    physical: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    uncertainties: dict[str, Any] = field(default_factory=dict)
    figure_links: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tic_id": self.tic_id,
            "target_id": self.target_id,
            "sample_id": self.sample_id,
            "candidate_id": self.candidate_id,
            "classification": self.classification,
            "confidence": self.confidence,
            "probability": self.probability,
            "transit": dict(self.transit),
            "physical": dict(self.physical),
            "quality": dict(self.quality),
            "uncertainties": dict(self.uncertainties),
            "figure_links": dict(self.figure_links),
        }


class CatalogBuilder:
    """Builds searchable catalogs in multiple formats."""

    def __init__(self, config: CatalogStageConfig) -> None:
        self.config = config

    def build(self, batch: ScientificInferenceBatch) -> list[CatalogEntry]:
        """Converts inference batch to catalog entries."""
        entries: list[CatalogEntry] = []
        for result in batch.results:
            if result.confidence < self.config.min_confidence:
                continue
            tic = result.target_id.replace("TIC ", "").strip()
            transit = result.transit.to_dict() if result.transit else {}
            physical = result.physical.to_dict() if result.physical else {}
            quality = (
                result.false_positive.to_dict() if result.false_positive else {}
            )
            uncertainties = (
                result.uncertainty.to_dict() if result.uncertainty else {}
            )
            figures = (
                result.explainability.to_dict() if result.explainability else {}
            )
            entries.append(
                CatalogEntry(
                    tic_id=tic,
                    target_id=result.target_id,
                    sample_id=result.sample_id,
                    candidate_id=result.candidate_id,
                    classification=result.classification,
                    confidence=result.confidence,
                    probability=result.probability,
                    transit=transit,
                    physical=physical,
                    quality=quality,
                    uncertainties=uncertainties,
                    figure_links={
                        k: v for k, v in figures.items() if v and k.endswith("_path")
                    },
                )
            )

        reverse = self.config.descending
        key_map = {
            "confidence": lambda e: e.confidence,
            "probability": lambda e: e.probability,
            "tic_id": lambda e: e.tic_id,
        }
        key_fn = key_map.get(self.config.sort_by, lambda e: e.confidence)
        entries.sort(key=key_fn, reverse=reverse)
        return entries

    def export(self, entries: list[CatalogEntry], output_dir: Path | str) -> dict[str, str]:
        """Writes catalog files in configured formats."""
        out = Path(output_dir)
        ensure_dir(out)
        records = [e.to_dict() for e in entries]
        flat_records = self._flatten_records(records)
        paths: dict[str, str] = {}

        if "json" in self.config.formats:
            json_path = out / f"{self.config.output_name}.json"
            write_json({"entries": records, "n_entries": len(records)}, json_path)
            paths["json"] = str(json_path)

        if "csv" in self.config.formats:
            csv_path = out / f"{self.config.output_name}.csv"
            pd.DataFrame(flat_records).to_csv(csv_path, index=False)
            paths["csv"] = str(csv_path)

        if "parquet" in self.config.formats:
            parquet_path = out / f"{self.config.output_name}.parquet"
            try:
                pd.DataFrame(flat_records).to_parquet(parquet_path, index=False)
                paths["parquet"] = str(parquet_path)
            except ImportError:
                logger.warning("Parquet export requires pyarrow; skipping parquet format.")

        return paths

    def _flatten_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        flat: list[dict[str, Any]] = []
        for rec in records:
            row = {
                "tic_id": rec["tic_id"],
                "target_id": rec["target_id"],
                "sample_id": rec["sample_id"],
                "candidate_id": rec["candidate_id"],
                "classification": rec["classification"],
                "confidence": rec["confidence"],
                "probability": rec["probability"],
            }
            for section in ("transit", "physical", "quality", "uncertainties"):
                for key, value in rec.get(section, {}).items():
                    if isinstance(value, (dict, list)):
                        row[f"{section}_{key}"] = json.dumps(value)
                    else:
                        row[f"{section}_{key}"] = value
            for key, value in rec.get("figure_links", {}).items():
                row[f"figure_{key}"] = value
            flat.append(row)
        return flat
