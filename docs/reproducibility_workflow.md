# Reproducibility Workflow

End-to-end workflow for reproducible research campaigns using the experiment orchestration system.

## 1. Register and run

```bash
exodet experiment -c configs/experiments.yaml
```

Each run captures:

- Git commit hash
- Python and library versions (NumPy, SciPy, PyTorch, scikit-learn, XGBoost)
- Hardware profile (CPU, RAM, CUDA)
- Configuration and dataset checksums
- Model checkpoint checksum

## 2. Sweep and compare

```bash
exodet sweep -c configs/experiments.yaml -o sweep.enabled=true
exodet leaderboard -c configs/experiments.yaml
```

Leaderboards export CSV, Markdown, and LaTeX tables under `outputs/reports/experiments/leaderboards/`.

## 3. Validate reproducibility

Enable the `reproduce` block:

```yaml
reproduce:
  enabled: true
  experiment_ids:
    - abc123def4567890
  metric_tolerance: 1.0e-4
  probability_tolerance: 1.0e-5
  issue_certificate: true
```

Run:

```bash
exodet reproduce -c configs/experiments.yaml
python scripts/reproduce.py -c configs/experiments.yaml
```

## 4. Certificates

Reproducibility certificates are written to `outputs/experiments/certificates/` with:

- Metric match verification (within tolerance)
- Model checksum consistency
- Prediction consistency (probability tolerance)
- SHA-256 signature over the certificate payload

## 5. Failure recovery

Interrupted experiments persist state in `state.json`. Checkpoints are verified via `verify_checkpoint()`. Resume sweeps with `sweep.resume: true`.

## 6. Artifact management

Artifacts are organized into category subdirectories (`models/`, `figures/`, `reports/`, etc.). Optional cleanup policies remove stale failed runs:

```yaml
artifacts:
  cleanup:
    enabled: true
    max_age_days: 30
    keep_status: [completed]
```

## Related documentation

- [experiments.md](experiments.md) — experiment manager
- [reproducibility.md](reproducibility.md) — snapshot collector
- [statistics.md](statistics.md) — statistical methods
