"""Incremental dataset growth and checkpoint-resume training."""

from exodet.update.checkpoint_manager import (
    CheckpointDiscovery,
    discover_checkpoint,
    list_experiment_checkpoints,
)
from exodet.update.config import UpdateStageConfig, load_update_stage_config
from exodet.update.dataset_registry import DatasetRegistry, TargetRecord
from exodet.update.resume import apply_training_resume, checkpoint_extra_state
from exodet.update.runner import run_update
from exodet.update.update_pipeline import (
    UpdateInputs,
    UpdatePipeline,
    download_tic_batch,
    merge_catalog_entries,
    merge_tce_catalog,
    parse_tic_ids_from_file,
    resolve_update_inputs,
)
from exodet.update.versioning import (
    DatasetManifest,
    append_to_splits,
    load_or_create_manifest,
)

__all__ = [
    "CheckpointDiscovery",
    "DatasetManifest",
    "DatasetRegistry",
    "TargetRecord",
    "UpdateInputs",
    "UpdatePipeline",
    "UpdateStageConfig",
    "append_to_splits",
    "apply_training_resume",
    "checkpoint_extra_state",
    "discover_checkpoint",
    "download_tic_batch",
    "list_experiment_checkpoints",
    "load_or_create_manifest",
    "load_update_stage_config",
    "merge_catalog_entries",
    "merge_tce_catalog",
    "parse_tic_ids_from_file",
    "resolve_update_inputs",
    "run_update",
]
