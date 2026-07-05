# Scientific Benchmarking

The `exodet benchmark` command runs a publication-oriented evaluation suite over classical baselines and optional neural models, using the same representation datasets and YAML configuration as training.

## Usage

```bash
exodet benchmark -c configs/benchmark.yaml
python scripts/run_benchmark.py -c configs/benchmark.yaml
```

## YAML block (`benchmark`)

| Key | Description |
|-----|-------------|
| `models` | Baseline registry names (`random_forest`, `xgboost`, `logistic_regression`, `mlp`, `lightgbm`) |
| `metrics` | Metric names passed to `compute_all_metrics` |
| `statistics` | Bootstrap/McNemar settings (`n_bootstrap`) |
| `calibration` | Reliability diagrams, ECE, MCE, Brier score |
| `error_analysis` | FP/FN stratification by period, depth, SNR, sector |
| `cross_mission` | Kepler/K2/TESS transfer metrics |
| `reports.formats` | `json`, `markdown`, `html`, `csv`, `pdf` |

## Outputs

Reports are written under `paths.report_dir/benchmark/`:

- `benchmark_manifest.json` — full payload + report paths
- `figures/` — ROC, PR, confusion matrices, calibration, error rates
- `checkpoints/<model>/` — fitted sklearn models

## Sensitivity

Enable the `sensitivity` block or run `exodet sensitivity`. Perturbations include Gaussian/red noise, missing cadences, period/epoch offsets, depth/duration scaling, and stellar variability. Performance curves are saved as `sensitivity_report.json`.

## Hyperparameter study

Set `hyperparameter.enabled: true` with a `parameters` grid. Trials are ranked by `ranking_metric` (default `roc_auc`).

## Statistical tests

Pairwise model comparisons use McNemar's test, bootstrap accuracy CIs, paired *t*-tests on probabilities, and Wilcoxon signed-rank tests. See [statistics.md](statistics.md).
