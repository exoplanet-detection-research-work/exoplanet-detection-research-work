# Hyperparameter Sweeps

Configure sweeps under the YAML `sweep` block and run with:

```bash
exodet sweep -c configs/experiments.yaml -o sweep.enabled=true
python scripts/run_sweep.py -c configs/experiments.yaml
```

## Methods

| Method | Description |
|--------|-------------|
| `grid` | Full Cartesian product over `parameters` |
| `random` | Uniform random sampling (`random_samples` trials) |
| `optuna` | TPE-style search (requires `optuna`; falls back to random) |
| `pbt` | Population-based sampling (`pbt.population_size`) |

## Configuration

```yaml
sweep:
  enabled: true
  method: grid
  model_name: logistic_regression
  ranking_metric: roc_auc
  max_trials: 0          # 0 = unlimited (grid) or use random_samples
  random_samples: 20
  resume: true
  parameters:
    C: [0.1, 1.0, 10.0]
    max_iter: [200, 500]
```

## Outputs

Each sweep writes to `outputs/experiments/sweeps/<sweep_id>/`:

- `sweep_result.json` — ranked trials with metrics and hyperparameter importance
- `sweep_state.json` — resume checkpoint for interrupted sweeps
- `tables/` — CSV, Markdown, and LaTeX leaderboards

## Hyperparameter importance

Correlation-based importance scores are computed post-sweep and included in `sweep_result.json` under `hyperparameter_importance`.

## Integration

Sweep trials are registered as individual experiments in the index, enabling `exodet leaderboard` comparisons across campaigns.
