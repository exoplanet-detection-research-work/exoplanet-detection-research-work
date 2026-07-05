"""Experiment registration, inheritance, and lifecycle management."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exodet.config.schema import ExperimentConfig
from exodet.exceptions import PipelineError
from exodet.experiments.database import ExperimentDatabase, ExperimentRecord
from exodet.experiments.profiling import ProfileContext, collect_hardware_profile
from exodet.experiments.templates import apply_template
from exodet.reproducibility.collector import checksum_file
from exodet.utils.io import ensure_dir, write_json

__all__ = ["ExperimentManager"]

logger = logging.getLogger(__name__)


def _generate_experiment_id(name: str, seed: int) -> str:
    digest = hashlib.sha256(f"{name}:{seed}:{uuid.uuid4()}".encode()).hexdigest()
    return digest[:16]


class ExperimentManager:
    """Registers, tracks, and runs experiments with automatic output layout."""

    def __init__(
        self,
        experiment: ExperimentConfig,
        *,
        database: ExperimentDatabase,
        stage_config: Any,
        config_path: Path | None = None,
        raw_config: dict[str, Any] | None = None,
    ) -> None:
        self.experiment = experiment
        self.database = database
        self.stage_config = stage_config
        self.config_path = config_path
        self.raw_config = raw_config or {}
        root = Path(experiment.paths.output_dir) / "experiments"
        if stage_config.output_dir:
            root = Path(stage_config.output_dir)
        self.campaign_root = root
        ensure_dir(self.campaign_root)

    def register(
        self,
        *,
        experiment_id: str | None = None,
        name: str | None = None,
        tags: tuple[str, ...] | None = None,
        parent_id: str | None = None,
        template: str | None = None,
    ) -> ExperimentRecord:
        """Register a new experiment and allocate its output directory."""
        eid = experiment_id or _generate_experiment_id(
            name or self.experiment.experiment_name, self.experiment.seed
        )
        output_dir = self.campaign_root / eid
        ensure_dir(output_dir)

        parent_record = None
        pid = parent_id or self.stage_config.parent_id
        if pid:
            parent_record = self.database.get(pid)
            if parent_record is None:
                raise PipelineError(f"Parent experiment '{pid}' not found.")
            if self.stage_config.inherit_config and parent_record.metadata.get("config"):
                self.raw_config = {**parent_record.metadata["config"], **self.raw_config}

        tpl = template or self.stage_config.template
        if tpl:
            self.raw_config = apply_template(tpl, self.raw_config)

        config_checksum = checksum_file(self.config_path) if self.config_path else ""
        hw = collect_hardware_profile(output_dir=output_dir)
        record = ExperimentRecord(
            experiment_id=eid,
            name=name or self.experiment.experiment_name,
            status="pending",
            tags=tags or self.stage_config.tags,
            parent_id=pid,
            template=tpl,
            config_path=str(self.config_path) if self.config_path else None,
            config_checksum=config_checksum,
            output_dir=str(output_dir),
            hardware=hw.to_dict(),
            git_commit=hw.git_commit,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={
                **self.stage_config.metadata,
                "seed": self.experiment.seed,
                "stage": self.stage_config.stage,
            },
        )
        if parent_record is not None:
            record.metadata["parent_name"] = parent_record.name
        write_json(record.to_dict(), output_dir / "experiment.json")
        self.database.register(record)
        logger.info("Registered experiment %s (%s)", eid, record.name)
        return record

    def mark_running(self, experiment_id: str) -> ExperimentRecord:
        return self.database.update(experiment_id, status="running")

    def mark_completed(
        self,
        experiment_id: str,
        *,
        metrics: dict[str, float],
        artifacts: dict[str, str],
        runtime_seconds: float,
        dataset_checksum: str = "",
        model_checksum: str = "",
    ) -> ExperimentRecord:
        record = self.database.update(
            experiment_id,
            status="completed",
            metrics=metrics,
            artifacts=artifacts,
            runtime_seconds=runtime_seconds,
            dataset_checksum=dataset_checksum,
            model_checksum=model_checksum,
            error=None,
        )
        write_json(record.to_dict(), Path(record.output_dir) / "experiment.json")
        return record

    def mark_failed(self, experiment_id: str, error: str) -> ExperimentRecord:
        return self.database.update(experiment_id, status="failed", error=error)

    def mark_interrupted(self, experiment_id: str) -> ExperimentRecord:
        return self.database.update(experiment_id, status="interrupted")

    def get_output_paths(self, experiment_id: str) -> dict[str, Path]:
        """Standard artifact subdirectories for an experiment."""
        record = self.database.get(experiment_id)
        if record is None:
            raise PipelineError(f"Unknown experiment id: {experiment_id}")
        root = Path(record.output_dir)
        paths = {
            "root": root,
            "checkpoints": root / "checkpoints",
            "figures": root / "figures",
            "reports": root / "reports",
            "logs": root / "logs",
            "catalogs": root / "catalogs",
            "benchmarks": root / "benchmarks",
            "datasets": root / "datasets",
        }
        for path in paths.values():
            ensure_dir(path)
        return paths

    def save_state(self, experiment_id: str, state: dict[str, Any]) -> Path:
        """Persist resumable run state."""
        record = self.database.get(experiment_id)
        if record is None:
            raise PipelineError(f"Unknown experiment id: {experiment_id}")
        path = Path(record.output_dir) / "state.json"
        write_json(state, path)
        return path

    def load_state(self, experiment_id: str) -> dict[str, Any] | None:
        record = self.database.get(experiment_id)
        if record is None:
            return None
        path = Path(record.output_dir) / "state.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def run_stage(
        self,
        experiment_id: str,
        *,
        stage: str | None = None,
    ) -> ExperimentRecord:
        """Execute the configured pipeline stage for a registered experiment."""
        from exodet.experiments.runner import execute_stage

        record = self.database.get(experiment_id)
        if record is None:
            raise PipelineError(f"Unknown experiment id: {experiment_id}")
        stage_name = stage or self.stage_config.stage
        self.mark_running(experiment_id)
        paths = self.get_output_paths(experiment_id)
        try:
            with ProfileContext(output_dir=paths["root"]) as profile:
                result = execute_stage(
                    self.experiment,
                    stage_name,
                    output_dir=paths["root"],
                    checkpoint_dir=paths["checkpoints"],
                )
            return self.mark_completed(
                experiment_id,
                metrics=result.get("metrics", {}),
                artifacts=result.get("artifacts", {}),
                runtime_seconds=profile.elapsed_seconds,
                dataset_checksum=result.get("dataset_checksum", ""),
                model_checksum=result.get("model_checksum", ""),
            )
        except Exception as exc:
            logger.exception("Experiment %s failed", experiment_id)
            self.mark_failed(experiment_id, str(exc))
            raise
