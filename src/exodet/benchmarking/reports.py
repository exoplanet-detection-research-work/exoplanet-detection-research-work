"""Benchmark report generation (HTML, PDF, Markdown, CSV, JSON)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exodet.inference.scientific import build_reproduction_metadata
from exodet.utils.io import ensure_dir, write_json

__all__ = ["BenchmarkReport", "write_benchmark_reports"]


@dataclass
class BenchmarkReport:
    """Aggregated benchmark report payload."""

    experiment_name: str
    dataset_summary: dict[str, Any]
    training_configuration: dict[str, Any]
    model_results: list[dict[str, Any]]
    statistics: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    error_analysis: dict[str, Any] = field(default_factory=dict)
    cross_mission: dict[str, Any] = field(default_factory=dict)
    sensitivity: dict[str, Any] = field(default_factory=dict)
    hyperparameter: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    conclusions: list[str] = field(default_factory=list)
    reproduction: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "dataset_summary": self.dataset_summary,
            "training_configuration": self.training_configuration,
            "model_results": self.model_results,
            "statistics": self.statistics,
            "calibration": self.calibration,
            "error_analysis": self.error_analysis,
            "cross_mission": self.cross_mission,
            "sensitivity": self.sensitivity,
            "hyperparameter": self.hyperparameter,
            "runtime": self.runtime,
            "hardware": self.hardware,
            "conclusions": self.conclusions,
            "reproduction": self.reproduction,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }


def _write_markdown(report: BenchmarkReport, path: Path) -> None:
    lines = [
        f"# Benchmark Report — {report.experiment_name}",
        "",
        "## Dataset",
        "",
        json.dumps(report.dataset_summary, indent=2),
        "",
        "## Model Results",
        "",
    ]
    for row in report.model_results:
        lines.append(f"### {row.get('name', 'model')}")
        lines.append("")
        lines.append(json.dumps(row.get("metrics", {}), indent=2))
        lines.append("")
    if report.statistics:
        lines.extend(["## Statistical Significance", "", json.dumps(report.statistics, indent=2), ""])
    if report.conclusions:
        lines.extend(["## Conclusions", ""])
        lines.extend(f"- {c}" for c in report.conclusions)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_html(report: BenchmarkReport, path: Path) -> None:
    body = [
        "<html><head><meta charset='utf-8'><title>Benchmark Report</title>",
        "<style>body{font-family:system-ui;max-width:960px;margin:2rem auto}",
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:.4rem}</style>",
        "</head><body>",
        f"<h1>Benchmark Report — {report.experiment_name}</h1>",
        "<h2>Model Metrics</h2><table><tr><th>Model</th><th>Metrics</th></tr>",
    ]
    for row in report.model_results:
        body.append(
            f"<tr><td>{row.get('name')}</td><td><pre>{json.dumps(row.get('metrics', {}), indent=2)}</pre></td></tr>"
        )
    body.append("</table>")
    if report.conclusions:
        body.append("<h2>Conclusions</h2><ul>")
        body.extend(f"<li>{c}</li>" for c in report.conclusions)
        body.append("</ul>")
    body.append("</body></html>")
    path.write_text("\n".join(body), encoding="utf-8")


def _write_csv_summary(report: BenchmarkReport, path: Path) -> None:
    metric_keys: set[str] = set()
    for row in report.model_results:
        metric_keys.update(row.get("metrics", {}).keys())
    columns = ["model", *sorted(metric_keys), "runtime_seconds"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in report.model_results:
            record = {"model": row.get("name"), "runtime_seconds": row.get("runtime_seconds")}
            record.update(row.get("metrics", {}))
            writer.writerow(record)


def _write_pdf(report: BenchmarkReport, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.1, 0.95, f"Benchmark Report — {report.experiment_name}", fontsize=16, weight="bold")
        y = 0.88
        for row in report.model_results:
            fig.text(0.1, y, f"{row.get('name')}: {json.dumps(row.get('metrics', {}))}", fontsize=9)
            y -= 0.06
            if y < 0.1:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(8.5, 11))
                y = 0.95
        if report.conclusions:
            fig.text(0.1, y, "Conclusions:", fontsize=11, weight="bold")
            y -= 0.04
            for conclusion in report.conclusions:
                fig.text(0.12, y, f"• {conclusion}", fontsize=9)
                y -= 0.04
        pdf.savefig(fig)
        plt.close(fig)


def write_benchmark_reports(
    report: BenchmarkReport,
    output_dir: Path,
    *,
    formats: tuple[str, ...] = ("json", "markdown", "html", "csv", "pdf"),
) -> dict[str, str]:
    """Write benchmark report in requested formats."""
    ensure_dir(output_dir)
    paths: dict[str, str] = {}
    stem = f"{report.experiment_name}_benchmark"
    if "json" in formats:
        p = output_dir / f"{stem}.json"
        write_json(report.to_dict(), p)
        paths["json"] = str(p)
    if "markdown" in formats:
        p = output_dir / f"{stem}.md"
        _write_markdown(report, p)
        paths["markdown"] = str(p)
    if "html" in formats:
        p = output_dir / f"{stem}.html"
        _write_html(report, p)
        paths["html"] = str(p)
    if "csv" in formats:
        p = output_dir / f"{stem}.csv"
        _write_csv_summary(report, p)
        paths["csv"] = str(p)
    if "pdf" in formats:
        p = output_dir / f"{stem}.pdf"
        _write_pdf(report, p)
        paths["pdf"] = str(p)
    return paths
