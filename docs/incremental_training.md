# Incremental training workflow

ExoDet supports automated dataset growth and checkpoint-resume training through the `update` stage. Researchers can supply new TIC IDs or local light curves; the system runs the full scientific pipeline, appends samples to existing splits, resumes training, and optionally evaluates the updated model.

## Quick start

```bash
# Single TIC ID
exodet update --tic 123456789

# CSV/TXT/JSON file of TIC IDs
exodet update --tic-file new_targets.csv

# Local FITS or processed NPZ directories
exodet update --fits ./new_lightcurves/
exodet update --processed ./new_processed/

# Resume training from latest or best checkpoint
exodet update --resume latest
exodet update --resume best

# Force reprocessing of already-registered targets
exodet update --force-reprocess
```

Configuration lives in `configs/update.yaml`. Override any setting with dotted keys:

```bash
exodet update -c configs/update.yaml -o update.parallel_workers=8
```

## Pipeline stages

For each new target the update orchestrator executes:

1. **Download** (TIC IDs only) — parallel fetch via lightkurve (TESS/Kepler/K2) with synthetic fallback for offline use
2. **Preprocessing** — existing preprocessing pipeline (detrend, sigma clip, normalize, …)
3. **TCE search** — BLS candidate generation and validation
4. **Representation** — phase folding, global/local views, physics features
5. **Dataset append** — append-only merge into configured split (default: `train`)
6. **Registry update** — record provenance in `dataset_registry.json`

Stage-level checkpoints are stored under `<interim_dir>/update_state/` so interrupted runs resume from the last completed stage.

## Training resume

When `update.resume_training: true`, the runner:

1. Discovers checkpoints under `outputs/checkpoints/<experiment_name>/`
2. Sets `training.trainer.params.resume_from` for the existing trainer
3. Restores optimizer, scheduler, AMP scaler, EMA/SWA, and curriculum state from checkpoint metadata
4. Continues training without reinitializing weights unless `--fresh-start` is passed

## Experiment provenance

Register incremental runs through the experiment manager (`experiments` YAML block):

- `experiment_mode: child` — new experiment linked to parent via `parent_id`
- `experiment_mode: continuation` — reuses the latest experiment record for the same name

## Python API

```python
from exodet.update import run_update, UpdatePipeline, resolve_update_inputs, UpdateStageConfig

payload = run_update(
    "configs/update.yaml",
    tic_ids=["123456789"],
    resume="best",
)

# Lower-level control
from exodet.update.config import load_update_stage_config

experiment, update_cfg, tce_cfg, rep_cfg, _ = load_update_stage_config("configs/update.yaml")
inputs = resolve_update_inputs(update_cfg, cli_tic_ids=["123456789"])
pipeline = UpdatePipeline(experiment, update_cfg, tce_cfg, rep_cfg)
summary = pipeline.run(inputs)
```

## Failure recovery

The update pipeline survives network interruption, partial downloads, corrupted FITS, failed preprocessing/BLS, GPU interruption, and keyboard interrupts. Per-target `update_state/<tic>.json` records completed stages; rerunning the same command skips finished work automatically.

## Outputs

| Artifact | Location |
|----------|----------|
| Dataset registry | `<processed_dir>/dataset_registry.json` |
| Version manifest | `<processed_dir>/dataset/manifest.json` |
| Update summary | `<report_dir>/update_summary.json` |
| Stage checkpoints | `<interim_dir>/update_state/` |

See also: [dataset versioning](dataset_versioning.md).
