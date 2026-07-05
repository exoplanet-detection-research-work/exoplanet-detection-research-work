# Experiment Orchestration

The `exodet experiment` command registers and runs managed experiments with automatic output directories, metadata tracking, and index persistence.

## Usage

```bash
exodet experiment -c configs/experiments.yaml
python scripts/run_experiment.py -c configs/experiments.yaml
```

## YAML block (`experiments`)

| Key | Description |
|-----|-------------|
| `database_path` | Experiment index JSON (default: `outputs/experiments/index.json`) |
| `tags` | Searchable tags attached to each run |
| `template` | Named template from `EXPERIMENT_TEMPLATES` |
| `parent_id` | Inherit configuration from a parent experiment |
| `stage` | Pipeline stage to execute (`train`, `evaluate`, `benchmark`, etc.) |
| `metadata` | Arbitrary metadata stored in the index |

## Experiment IDs

Each run receives a unique 16-character hex ID. Output is written to:

```
outputs/experiments/<experiment_id>/
  experiment.json
  checkpoints/
  figures/
  reports/
  run_summary.json
```

## Templates

Built-in templates (see `exodet.experiments.templates`):

- `cnn_baseline`, `transformer_baseline`, `hybrid_model`
- `ablation_study`, `calibration_study`, `sensitivity_study`
- `cross_mission_eval`, `sklearn_baseline`

## Database

The experiment index stores configuration checksums, metrics, artifacts, hardware profile, git commit, dataset/model checksums, runtime, and status. Search via `ExperimentDatabase.search()`.

## Related commands

- `exodet sweep` — hyperparameter campaigns
- `exodet leaderboard` — ranked comparisons
- `exodet reproduce` — reproducibility validation

See also [hyperparameter_sweeps.md](hyperparameter_sweeps.md) and [reproducibility_workflow.md](reproducibility_workflow.md).
