"""Publication-ready table generation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from exodet.experiments.comparison import ComparisonReport, Leaderboard
from exodet.experiments.database import ExperimentRecord
from exodet.utils.io import ensure_dir

__all__ = [
    "write_csv_table",
    "write_markdown_table",
    "write_latex_table",
    "export_publication_tables",
]


def write_csv_table(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def write_latex_table(
    rows: list[dict[str, Any]],
    columns: list[str],
    *,
    caption: str = "Experiment results",
    label: str = "tab:experiments",
) -> str:
    col_spec = "l" + "r" * (len(columns) - 1)
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(columns) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(row.get(c, "")) for c in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def export_publication_tables(
    report: ComparisonReport,
    output_dir: Path,
    *,
    formats: tuple[str, ...] = ("csv", "markdown", "latex"),
) -> dict[str, str]:
    """Export leaderboards as publication tables."""
    ensure_dir(output_dir)
    paths: dict[str, str] = {}
    for lb in report.leaderboards:
        if not lb.rows:
            continue
        columns = ["rank", "name", "experiment_id", "metric_value", "runtime_seconds"]
        stem = f"leaderboard_{lb.metric}"
        if "csv" in formats:
            p = output_dir / f"{stem}.csv"
            write_csv_table(lb.rows, p, columns)
            paths[f"{stem}_csv"] = str(p)
        if "markdown" in formats:
            p = output_dir / f"{stem}.md"
            p.write_text(write_markdown_table(lb.rows, columns), encoding="utf-8")
            paths[f"{stem}_md"] = str(p)
        if "latex" in formats:
            p = output_dir / f"{stem}.tex"
            p.write_text(
                write_latex_table(lb.rows, columns, caption=f"Leaderboard ({lb.metric})"),
                encoding="utf-8",
            )
            paths[f"{stem}_tex"] = str(p)
    return paths


def records_to_table(records: list[ExperimentRecord], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    """Convert experiment records to flat table rows."""
    rows: list[dict[str, Any]] = []
    for rec in records:
        row: dict[str, Any] = {
            "experiment_id": rec.experiment_id,
            "name": rec.name,
            "status": rec.status,
            "runtime_seconds": rec.runtime_seconds,
        }
        for m in metrics:
            row[m] = rec.metrics.get(m, "")
        rows.append(row)
    return rows
