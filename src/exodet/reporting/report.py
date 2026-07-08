"""Candidate report generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exodet.inference.containers import ScientificInferenceResult
from exodet.inference.config import ReportStageConfig
from exodet.representation.containers import DatasetSample
from exodet.utils.io import ensure_dir, write_json
from exodet.utils.paths import safe_filename

__all__ = ["CandidateReport", "ReportGenerator"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CandidateReport:
    """Structured report for one candidate."""

    result: ScientificInferenceResult
    sample: DatasetSample
    figure_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "result": self.result.to_dict(),
            "sample": {
                "sample_id": self.sample.sample_id,
                "target_id": self.sample.target_id,
                "label": int(self.sample.label),
            },
            "figure_paths": dict(self.figure_paths),
        }


class ReportGenerator:
    """Builds JSON, CSV, and PDF candidate reports."""

    def __init__(self, config: ReportStageConfig) -> None:
        self.config = config

    def generate(
        self,
        result: ScientificInferenceResult,
        sample: DatasetSample,
        output_dir: Path,
    ) -> CandidateReport:
        """Generates report artefacts for one candidate."""
        ensure_dir(output_dir)
        prefix = safe_filename(result.sample_id)
        figures = self._make_figures(result, sample, output_dir, prefix)

        if "json" in self.config.formats:
            write_json(self.to_dict(result, sample, figures), output_dir / f"{prefix}_report.json")

        if "csv" in self.config.formats:
            pd.DataFrame([self._flat_row(result)]).to_csv(
                output_dir / f"{prefix}_summary.csv", index=False
            )

        if "pdf" in self.config.formats:
            self._write_pdf(result, sample, figures, output_dir / f"{prefix}_report.pdf")

        return CandidateReport(result=result, sample=sample, figure_paths=figures)

    def _make_figures(
        self,
        result: ScientificInferenceResult,
        sample: DatasetSample,
        output_dir: Path,
        prefix: str,
    ) -> dict[str, str]:
        paths: dict[str, str] = {}
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        axes[0, 0].plot(sample.global_view, color="0.2")
        axes[0, 0].set_title("Global phase-folded view")
        axes[0, 0].set_xlabel("Orbital phase bin")
        axes[0, 0].set_ylabel("Relative flux (normalized)")
        axes[0, 0].set_xlabel("Orbital phase bin")

        axes[0, 1].plot(sample.local_view, color="0.3")
        axes[0, 1].set_title("Local transit view")
        axes[0, 1].set_xlabel("Orbital phase bin")
        axes[0, 1].set_ylabel("Relative flux (normalized)")

        if result.transit is not None and self.config.include_transit_fit:
            phase = np.linspace(-0.5, 0.5, len(sample.local_view), endpoint=False)
            depth = result.transit.depth
            hw = result.transit.duration_days / (2.0 * result.transit.period_days)
            model = 1.0 - depth * (np.abs(phase) <= hw)
            axes[1, 0].plot(sample.local_view, label="data", color="0.3")
            axes[1, 0].plot(phase, model, label="fit", color="crimson", ls="--")
            axes[1, 0].legend()
            axes[1, 0].set_xlabel("Orbital phase")
            axes[1, 0].set_ylabel("Relative flux (normalized)")
            axes[1, 0].set_title("Transit fit (trapezoid)")
        else:
            axes[1, 0].axis("off")

        axes[1, 1].bar(
            ["P(planet)", "confidence", "fp_risk"],
            [
                result.probability,
                result.confidence,
                result.false_positive.overall_fp_risk if result.false_positive else 0.0,
            ],
            color=["steelblue", "seagreen", "darkorange"],
        )
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title("Classification summary")

        fig.suptitle(f"{result.target_id} — {result.classification}")
        fig.tight_layout()
        overview = output_dir / f"{prefix}_overview.png"
        fig.savefig(overview, dpi=self.config.figure_dpi)
        plt.close(fig)
        paths["overview"] = str(overview)

        if result.explainability is not None:
            for key, path in result.explainability.to_dict().items():
                if path:
                    paths[key.replace("_path", "")] = path

        return paths

    def _flat_row(self, result: ScientificInferenceResult) -> dict[str, Any]:
        row: dict[str, Any] = {
            "sample_id": result.sample_id,
            "target_id": result.target_id,
            "classification": result.classification,
            "probability": result.probability,
            "confidence": result.confidence,
        }
        if result.transit is not None:
            row.update(result.transit.to_dict())
        if result.physical is not None:
            row.update(result.physical.to_dict())
        if result.uncertainty is not None:
            row.update({f"uncertainty_{k}": v for k, v in result.uncertainty.to_dict().items()})
        return row

    def to_dict(
        self,
        result: ScientificInferenceResult,
        sample: DatasetSample,
        figures: dict[str, str],
    ) -> dict[str, Any]:
        return CandidateReport(result=result, sample=sample, figure_paths=figures).to_dict()

    def _write_pdf(
        self,
        result: ScientificInferenceResult,
        sample: DatasetSample,
        figures: dict[str, str],
        path: Path,
    ) -> None:
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(path) as pdf:
            if "overview" in figures:
                img = plt.imread(figures["overview"])
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(f"Candidate report: {result.target_id}")
                pdf.savefig(fig)
                plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis("off")
            lines = [
                f"Target: {result.target_id}",
                f"Classification: {result.classification}",
                f"Probability: {result.probability:.4f}",
                f"Confidence: {result.confidence:.4f}",
            ]
            if result.transit is not None:
                t = result.transit
                lines.extend(
                    [
                        f"Depth: {t.depth:.5f}",
                        f"Period: {t.period_days:.5f} d",
                        f"Duration: {t.duration_days:.5f} d",
                        f"Rp/Rs: {t.rp_rs:.4f}",
                    ]
                )
            if result.physical is not None:
                p = result.physical
                if p.planet_radius_rearth is not None:
                    lines.append(f"Rp (R_Earth): {p.planet_radius_rearth:.2f}")
                if p.semi_major_axis_au is not None:
                    lines.append(f"a (AU): {p.semi_major_axis_au:.4f}")
            ax.text(0.05, 0.95, "\n".join(lines), va="top", family="monospace")
            pdf.savefig(fig)
            plt.close(fig)
