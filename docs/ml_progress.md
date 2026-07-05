# Deep learning infrastructure — implementation progress

Working notes for the ML training ecosystem (`exodet.ml`).

## Status: COMPLETE

All 14 modules implemented, tested, and integrated.

### Package layout (`src/exodet/ml/`)

| Module | File | Status |
|--------|------|--------|
| Model registry | `models.py` | ✓ BaseTorchModel, XGBoostModel, MODEL_BACKENDS, ARCHITECTURE_KINDS |
| Training engine | `trainer.py` | ✓ SupervisedTrainer → TRAINERS `supervised` |
| Loss registry | `losses.py` | ✓ bce, weighted_bce, focal, label_smooth_bce |
| Optimizer registry | `optimizers.py` | ✓ adamw, adam, sgd, rmsprop, lion |
| Scheduler registry | `schedulers.py` | ✓ cosine, warm_restarts, one_cycle, plateau, linear_warmup |
| Metrics | `metrics.py` | ✓ 10 metrics → METRICS registry |
| Checkpoints | `checkpoints.py` | ✓ best/last/top-k, full state resume |
| Mixed precision | `amp.py` | ✓ FP16/BF16, GradScaler, CPU/MPS fallback |
| Callbacks | `callbacks.py` | ✓ early_stopping, checkpoint, lr_monitor, grad_clip, predict_export |
| Experiment tracking | `tracking.py` | ✓ CSV, TensorBoard, optional W&B |
| Cross-validation | `cross_validation.py` | ✓ kfold, repeated, group/star, nested |
| Inference | `inference.py` | ✓ batch + single, calibration/uncertainty hooks |
| Data module | `data.py` | ✓ RepresentationDataModule, MlBatch |
| Config | `config.py` | ✓ MlSettings from training.trainer.params |
| Device | `device.py` | ✓ CPU/CUDA/MPS auto-select |
| Runner | `runner.py` | ✓ run_training, run_evaluation, run_predict |

### Integration

- CLI: `exodet train|evaluate|predict` wired in `cli/main.py`
- Config: `configs/train_example.yaml` (XGBoost sklearn path)
- Tests: `tests/test_ml.py`, `tests/test_ml_integration.py`, `tests/ml_fixtures.py`
- Benchmark: `scripts/benchmark_ml.py`
- Deps: `pip install exodet[ml]` adds torch + tensorboard

### Design notes

- No CNN/Transformer/attention/classifier implementations (next phase).
- `linear_probe` test model in `tests/ml_fixtures.py` only.
- Extended hyperparameters in `training.trainer.params` (no schema break).
- `torch_trainer` alias resolves to `supervised` in `build_trainer()`.

### Test results

268 tests passing (19 ML-specific + existing suite).
