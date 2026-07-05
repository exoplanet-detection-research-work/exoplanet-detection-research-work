# Architecture Ablation

The ablation framework systematically compares encoder combinations using registered architectures or sklearn proxies.

## Usage

```bash
exodet ablation -c configs/ablation.yaml
python scripts/run_ablation.py -c configs/ablation.yaml
```

## Variants

Default variants (configurable in `ablation.variants`):

| ID | Architecture | Description |
|----|--------------|-------------|
| `cnn_only` | `cnn_only` | CNN branch only |
| `transformer_only` | `transformer_only` | Transformer branch only |
| `physics_only` | `physics_only` | Physics MLP only |
| `cnn_transformer` | `cnn_transformer` | CNN + Transformer |
| `fusion` | `fusion` | Full hybrid (CNN + Transformer + Physics) |

> **Note:** `cnn_physics` and `transformer_physics` branch modes are not yet exposed in `ModelArchitectureConfig._BRANCHES`. The default config documents these rows; use `backend: sklearn` proxies or extend branch modes in a future release.

## Backend

- `backend: sklearn` — fast comparative baselines on flattened features (CI-friendly).
- `backend: torch` — trains registered neural architectures with `fast_training` overrides (bins, depth, epochs).

## Outputs

- `ablation_report.json` — per-variant status, metrics, runtime
- `ablation_table.json` — comparison table keyed by label
- `figures/ablation_summary.*` — bar chart of ranking metric
