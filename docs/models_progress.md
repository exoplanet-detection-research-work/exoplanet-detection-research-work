# Neural network architectures — implementation progress

Context for the hybrid CNN + Transformer exoplanet classifier (`exodet.models`).

## Status: COMPLETE

All 12 modules implemented, tested, benchmarked, and integrated with the existing
trainer/registry/CLI without modifying repository architecture.

## Package layout (`src/exodet/models/`)

| File | Module | Description |
|------|--------|-------------|
| `config.py` | 9 | `ModelArchitectureConfig` — every hyperparameter YAML-configurable |
| `cnn.py` | 1 | `LocalCNNEncoder` — residual/dw-sep CNN on `(B,1,401)` |
| `transformer.py` | 2 | `GlobalTransformerEncoder` — CLS pre-norm transformer on `(B,2001)` |
| `physics_encoder.py` | 3 | `PhysicsFeatureEncoder` — MLP on 25+ physics features |
| `fusion.py` | 4 | `CrossAttentionFusion` — cross-attn + gated + residual fusion |
| `classifier.py` | 5–7 | Heads + `HybridExoplanetNetwork` with cached `forward_features()` |
| `registry.py` | 8 | `ExoplanetClassifierModel` wrapper + MODELS registrations |
| `visualization.py` | 10 | CLS attention, CNN activations, PCA/t-SNE, feature importance |

## Registry entries (YAML `model.architecture.name`)

| Name | Branches |
|------|----------|
| `fusion` | CNN + Transformer + Physics |
| `cnn_transformer` | CNN + Transformer |
| `cnn_only` / `cnn` | Local CNN only |
| `transformer_only` / `transformer` | Global transformer only |
| `physics_only` | Physics MLP only |

## Class taxonomy (default `num_classes=5`)

0 transit · 1 eclipsing_binary · 2 variable_star · 3 blend · 4 noise

## Trainer integration (no Trainer changes)

- `forward_batch()` returns `(B,)` transit-vs-rest logit (`trainer_output: binary_transit`)
  for BCE compatibility with `SupervisedTrainer`
- Full multi-class `predict_proba_multiclass()` / `forward()` available on the module
- Import `exodet.models.registry` before `MODELS.build()` (done in `ml/runner.py`)

## Config & examples

- `configs/models.yaml` — architecture defaults per variant + benchmark sizes
- `configs/fusion_train_example.yaml` — end-to-end fusion training example

## Scripts & tests

- `scripts/benchmark_models.py` — forward+backward throughput per architecture
- `tests/test_models.py` — 33 parametrized tests (forward/backward, API, CUDA, TorchScript)

## Performance features

- Optional `torch.compile()` per branch
- Transformer gradient checkpointing (`transformer_checkpoint: true`)
- Vectorized attention/conv; no Python loops in forward
- Mixed precision via existing trainer `amp` setting (unchanged)

## Test results

301 total tests passing (33 new architecture tests).

## Next phase (out of scope here)

- Cross-entropy loss wiring when trainer label casting supports multi-class
- Temperature scaling / calibration on `ConfidenceHead`
- Self-supervised / ensemble methods
