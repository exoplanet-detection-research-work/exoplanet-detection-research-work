# Scientific inference layer — COMPLETE

Post-training inference, explainability, reporting, and catalog modules.

## Packages

| Package | Role |
|---------|------|
| `exodet.inference` | Scientific inference pipeline |
| `exodet.reporting` | Candidate reports (JSON/PDF/CSV) |
| `exodet.catalog` | Searchable catalogs (CSV/JSON/Parquet) |

## CLI

- `exodet infer` — scientific inference
- `exodet report` — candidate reports
- `exodet catalog` — catalog build
- `exodet compare` — multi-model comparison

## Modules (14)

| # | Module | Status |
|---|--------|--------|
| 1 | Inference pipeline | ✓ batch/single/directory/streaming |
| 2 | Transit parameter refinement | ✓ least squares + robust + bootstrap |
| 3 | Physical parameters | ✓ graceful missing metadata |
| 4 | Uncertainty | ✓ MC dropout, bootstrap, credible intervals |
| 5 | Explainability | ✓ Grad-CAM, IG, attention, occlusion |
| 6 | False positive analysis | ✓ 7 diagnostic scores |
| 7 | Scientific reports | ✓ JSON/PDF/CSV + figures |
| 8 | Catalog builder | ✓ CSV/JSON/Parquet |
| 9 | Model comparison | ✓ ROC/PR/McNemar/agreement |
| 10 | Inference benchmarking | ✓ latency + throughput |
| 11 | CLI integration | ✓ infer/report/catalog/compare |
| 12 | YAML configuration | ✓ inference/report/catalog blocks |
| 13 | Tests | ✓ 15 new tests |
| 14 | Documentation | ✓ 4 docs |

## Test status

- New tests: **15 passed**
- Full suite: **336 passed** (3 skipped)

## Integration notes

- `ml.inference.InferenceEngine` API unchanged
- Stage YAML keys (`inference`, `report`, `catalog`, `compare`) stripped before `ExperimentConfig` validation
- Wraps existing checkpoint layout (`best.pt`)
