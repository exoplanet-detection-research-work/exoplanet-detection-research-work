# Dataset versioning

ExoDet tracks incremental dataset changes through a versioned manifest and per-target registry. This ensures reproducibility when new TESS targets are added without rebuilding the full dataset.

## Registry (`dataset_registry.json`)

Located at `<processed_dir>/dataset_registry.json` by default (override with `update.registry_path`).

Each processed target records:

| Field | Description |
|-------|-------------|
| `tic_id` | Normalized TIC identifier |
| `target_id` | Canonical target string (e.g. `TIC 123456789`) |
| `mission` | Mission name (TESS, Kepler, K2) |
| `download_date` | ISO timestamp of acquisition |
| `sectors` | Observed sectors when available |
| `processing_version` | Pipeline version tag |
| `preprocessing_version` | Preprocessing config version |
| `tce_version` | TCE stage version |
| `phase_fold_version` | Representation folding version |
| `dataset_checksum` | SHA-256 of the split file after append |
| `dataset_split` | Split the samples were appended to |
| `sample_ids` | IDs of newly added samples |

Before processing, the pipeline checks the registry and skips duplicates unless `force_reprocess: true` or `--force-reprocess` is set.

## Manifest (`manifest.json`)

Located at `<processed_dir>/dataset/manifest.json`.

Tracks split-level versioning:

- `version` — dataset version string from `update.dataset_version` or representation config
- `n_samples` — sample count per split after each append
- `sample_ids` — full sample ID list per split
- `checksums` — SHA-256 of each split NPZ file
- `append_log` — chronological log of append operations with timestamps and added sample IDs

## Append-only semantics

New samples are merged by `sample_id`:

- Existing samples are never removed or modified
- Duplicate `sample_id` values are ignored
- Train/validation/test assignments of existing samples are preserved
- New samples append to the split configured by `update.append_split` (default: `train`)

To rebuild all splits from scratch, use the standard `exodet dataset` command with an explicit rebuild policy rather than the update stage.

## Feature scaler consistency

When appending samples, the update pipeline loads the existing `feature_scaler.json` from the dataset directory if present. New samples are transformed with the same scaler fitted on the original training split, preventing statistic leakage from new data into scaler refits.

If no scaler exists (first update on an empty dataset), a scaler is fit on the new samples and saved for subsequent appends.

## Traceability

Every append log entry includes:

```json
{
  "timestamp": "2026-07-05T12:00:00+00:00",
  "split": "train",
  "n_added": 3,
  "sample_ids": ["sample-a", "sample-b", "sample-c"]
}
```

Combined with the registry and representation sample provenance (`history`, `meta`), this provides end-to-end traceability from TIC ID to trained model checkpoint.

## Configuration

```yaml
update:
  dataset_version: v2
  append_split: train
  registry_path: data/processed/dataset_registry.json
  force_reprocess: false
```

## Python API

```python
from exodet.update import DatasetRegistry, append_to_splits, load_or_create_manifest

registry = DatasetRegistry("data/processed/dataset_registry.json")
if registry.should_process("123456789"):
    ...

manifest = load_or_create_manifest(
    Path("data/processed/dataset/manifest.json"),
    version="v2",
    experiment_name="exodet_incremental_v1",
)
```
