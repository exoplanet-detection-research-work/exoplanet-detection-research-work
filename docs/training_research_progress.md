# Research training strategy — COMPLETE

Publication-grade training extensions integrated with the existing `SupervisedTrainer` API.

## Integration (no Trainer API changes)

- `ResearchSupervisedTrainer` subclasses `SupervisedTrainer`, registered as `research` in `TRAINERS`
- YAML: `training.trainer.params.research` parsed by `ResearchTrainingConfig`
- Activate via `training.trainer.name: research` **or** `research.enabled: true`
- `build_trainer()` in `ml/trainer.py` routes to research trainer
- `ResearchDataModule` wraps curriculum, imbalance samplers, augmentation, hard mining
- Research callbacks register into `CALLBACKS` (`ema`, `swa`)

## Modules (14)

| # | Module | File | Status |
|---|--------|------|--------|
| 1 | Curriculum learning | `curriculum.py` | ✓ SNR stages, auto scheduling |
| 2 | Class imbalance | `curriculum.py` + `data.py` | ✓ weighted / balanced / effective-N |
| 3 | Augmentation | `augmentation.py` | ✓ 12+ scientifically valid ops |
| 4 | Hard example mining | `data.py`, `monitoring.py` | ✓ loss/confidence tracking |
| 5 | Distillation | `distillation.py` | ✓ soft labels, temperature, α blend |
| 6 | Masked pretraining | `pretraining.py` | ✓ random/patch masking, checkpoint export |
| 7 | Contrastive | `contrastive.py` | ✓ SimCLR, NT-Xent, projection head |
| 8 | Calibration | `calibration.py` | ✓ temperature scaling, ECE, reliability |
| 9 | Advanced evaluation | `evaluation.py` | ✓ ROC/PR/confusion/calibration + strata |
| 10 | Monitoring | `monitoring.py` | ✓ gradients, entropy, throughput |
| 11 | Checkpoint averaging | `checkpoint_averaging.py` | ✓ EMA, SWA, best-k ensemble |
| 12 | Scientific validation | `evaluation.py` | ✓ period/depth/noise strata tables |
| 13 | Benchmarking | `benchmarking.py` | ✓ CPU/CUDA/AMP matrix |
| 14 | Tests | `tests/test_training_research.py` | ✓ 20 tests |

## Deliverables

```
src/exodet/training/
  config.py, curriculum.py, augmentation.py, contrastive.py,
  pretraining.py, distillation.py, calibration.py, evaluation.py,
  monitoring.py, checkpoint_averaging.py, benchmarking.py,
  research_trainer.py, data.py, __init__.py
configs/training_research.yaml
scripts/pretrain.py, calibrate.py, evaluate_research.py, benchmark_training.py
tests/test_training_research.py
```

## Usage

```bash
# Research supervised training (via existing train stage)
exodet train -c configs/training_research.yaml

# Standalone scripts
python scripts/pretrain.py -c configs/training_research.yaml
python scripts/calibrate.py --checkpoint outputs/.../best.pt --config ...
python scripts/evaluate_research.py -c configs/training_research.yaml
python scripts/benchmark_training.py --device cpu
```

## Test status

- `tests/test_training_research.py`: 20 passed
- Full suite: **321 passed** (with ML extras installed)

## Key design notes

- Representation augmenters use `.apply()`; training augmenters use `__call__` — pipeline handles both
- `ResearchTrainingConfig` is frozen/slotted — use `dataclasses.asdict()` for callback builders
- Hard mining updates from training-batch losses each epoch
- Pretraining masks all view channels jointly; slices restored for multi-branch forward
