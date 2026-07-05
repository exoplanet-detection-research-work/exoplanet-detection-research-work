# Scientific inference

The `exodet.inference` package extends post-training analysis without modifying `ml.inference.InferenceEngine`.

## Pipeline

`ScientificInferencePipeline` supports:

- Batch inference on `RepresentationDataset`
- Single-target and streaming inference
- Directory inference (`*.npz` globs)
- Automatic checkpoint loading (`best.pt` or standalone `.pt`)
- CPU/CUDA fallback and mixed-precision batching

```bash
exodet infer -c configs/inference.yaml
python scripts/run_inference.py -c configs/inference.yaml
```

## Configuration

Stage settings live under the YAML `inference:` block (see `configs/inference.yaml`). The experiment schema (`data`, `model`, `training`, `evaluation`) is unchanged; stage keys are stripped before `ExperimentConfig` validation.

## Outputs

- `{report_dir}/scientific_inference.json` — full per-candidate results
- `{report_dir}/inference_summary.json` — run metadata
- Optional `{report_dir}/inference_benchmark.json`

## Architecture

| Module | Role |
|--------|------|
| `pipeline.py` | Orchestration |
| `parameter_fit.py` | Transit refinement |
| `physical.py` | Planetary parameters |
| `uncertainty.py` | MC dropout, bootstrap |
| `explainability.py` | Grad-CAM, IG, attention |
| `false_positive.py` | Astrophysical FP diagnostics |
| `comparison.py` | Multi-model statistical comparison |
| `benchmark.py` | Latency / throughput |

The base `InferenceEngine` API is reused for probability estimation; scientific layers are additive.
