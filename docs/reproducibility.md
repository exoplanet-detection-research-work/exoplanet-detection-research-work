# Reproducibility

`exodet reproducibility` captures everything required to reproduce a benchmark or training run.

## Usage

```bash
exodet reproducibility -c configs/benchmark.yaml
python scripts/run_reproducibility.py -c configs/benchmark.yaml
```

## Recorded metadata

| Field | Source |
|-------|--------|
| Git commit | `git rev-parse HEAD` |
| Python version | `sys.version` |
| Package versions | numpy, scipy, torch, sklearn, xgboost |
| Operating system | `platform.platform()` |
| Hardware | CPU, optional CUDA device, memory |
| Random seed | `experiment.seed` |
| Configuration snapshot | Model, training, checksum of YAML |
| Dataset checksums | SHA-256 of `train/val/test.npz` when present |
| Model checksums | SHA-256 of checkpoint artifacts |

## Outputs

Written to `paths.report_dir/reproducibility/`:

- `reproducibility.json` — machine-readable snapshot
- `reproducibility.md` — human-readable summary

Benchmark and ablation reports embed the same snapshot under `reproduction`.

## API

```python
from exodet.reproducibility import collect_reproducibility_snapshot, run_reproducibility

snapshot = collect_reproducibility_snapshot(experiment, config_path=Path("configs/benchmark.yaml"))
```
