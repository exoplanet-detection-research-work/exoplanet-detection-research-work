# ExoDet — Exoplanet Detection from Photometric Light Curves

> Config-driven, publication-quality pipeline for detecting transiting
> exoplanets in Kepler / K2 / TESS light curves with machine learning.

**Status:** architecture scaffold — pipeline interfaces are defined;
concrete data sources, preprocessing steps, and models land next.

## Overview

<!-- TODO: 2–3 paragraphs: scientific motivation, datasets used,
     modelling approach, headline results, link to paper/preprint. -->

## Repository layout

```
.
├── configs/                  # YAML experiment configs (base + experiments)
├── data/                     # Never committed (see .gitignore)
│   ├── raw/                  #   as-downloaded archive files
│   ├── interim/              #   intermediate preprocessing artifacts
│   ├── processed/            #   model-ready datasets
│   └── external/             #   third-party reference data
├── docs/                     # Sphinx documentation sources
├── notebooks/                # Exploratory analysis (not pipeline code)
├── outputs/
│   ├── checkpoints/          # saved model weights
│   ├── figures/              # generated figures (pdf + png)
│   ├── logs/                 # per-run log files
│   └── reports/              # evaluation reports (JSON)
├── scripts/                  # one-off maintenance scripts
├── src/exodet/
│   ├── cli/                  # `exodet` command-line entrypoint
│   ├── config/               # YAML schema + loader (inheritance, overrides)
│   ├── constants.py          # physical/astronomical/mission constants
│   ├── data/                 # LightCurve container, source/dataset ABCs
│   ├── evaluation/           # metric ABC, evaluation reports
│   ├── exceptions.py         # package exception hierarchy
│   ├── features/             # feature-extractor ABC
│   ├── models/               # model ABC (framework-agnostic)
│   ├── preprocessing/        # preprocessor ABC + pipeline composition
│   ├── registry.py           # name→class registries powering YAML configs
│   ├── training/             # trainer ABC + TrainingResult
│   ├── utils/                # logging, io, seeding, timing, validation
│   └── visualization/        # publication figure style helpers
└── tests/                    # pytest suite
```

## Installation

Requires Python 3.11+.

### pip (development install)

```bash
python -m venv .venv
```

#### Windows

```cmd
.venv\Scripts\activate
pip install -e ".[dev]"
```

#### macOS / Linux

```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

### conda

```bash
conda env create -f environment.yml
conda activate exodet
```

## Usage

Every command is driven by a YAML config; experiment configs inherit
from `configs/base.yaml` via the `defaults` key and can be tweaked from
the command line with repeated `--override` flags.

```bash
# Validate a config without running anything
exodet validate-config -c configs/experiment_example.yaml

# Show version + all registered pipeline components
exodet info

# Pipeline stages (available once concrete components are implemented)
exodet download   -c configs/experiment_example.yaml
exodet preprocess -c configs/experiment_example.yaml
exodet tce        -c configs/tce_example.yaml
exodet train      -c configs/experiment_example.yaml -o training.epochs=200
exodet evaluate   -c configs/experiment_example.yaml
exodet predict    -c configs/experiment_example.yaml
```

## Architecture

The pipeline is composed of pluggable component families, each defined
by an abstract base class and a registry:

| Family             | ABC                    | Registry             | Config section         |
| ------------------ | ---------------------- | -------------------- | ---------------------- |
| Data sources       | `BaseDataSource`       | `DATA_SOURCES`       | `data.source`          |
| Datasets           | `BaseDataset`          | `DATASETS`           | `data.dataset`         |
| Preprocessors      | `BasePreprocessor`     | `PREPROCESSORS`      | `preprocessing.steps`  |
| TCE search stages  | (per-stage classes)    | `GRID_GENERATORS`, `SEARCH_ENGINES`, `PEAK_DETECTORS`, `METRICS_COMPUTERS`, `VALIDATORS`, `HARMONIC_REJECTERS`, `RANKERS` | `configs/tce_example.yaml` |
| Feature extractors | `BaseFeatureExtractor` | `FEATURE_EXTRACTORS` | `model.features`       |
| Models             | `BaseModel`            | `MODELS`             | `model.architecture`   |
| Trainers           | `BaseTrainer`          | `TRAINERS`           | `training.trainer`     |
| Metrics            | `BaseMetric`           | `METRICS`            | `evaluation.metrics`   |

Data flows as immutable `LightCurve` objects that carry provenance
history through the preprocessing pipeline, then through feature
extraction into models.

### Extending the pipeline

Adding a component never requires touching existing code:

```python
from exodet.data.base import LightCurve
from exodet.preprocessing.base import PREPROCESSORS, BasePreprocessor


@PREPROCESSORS.register("sigma_clip")
class SigmaClipper(BasePreprocessor):
    """Removes outliers beyond N standard deviations."""

    def __init__(self, sigma: float = 5.0) -> None:
        self.sigma = sigma

    def apply(self, light_curve: LightCurve) -> LightCurve:
        ...
```

then reference it in YAML:

```yaml
preprocessing:
  steps:
    - name: sigma_clip
      params:
        sigma: 5.0
```

## Reproducibility

- Global seeding (`seed` config key) covers Python, NumPy, and — when
  installed — PyTorch/TensorFlow.
- Configs are immutable after loading; every run logs its resolved
  configuration and writes timestamped log files to `outputs/logs/`.
- Raw data integrity is verified with SHA-256 checksums.

## Development

```bash
pytest                 # run test suite
ruff check src tests   # lint
black src tests        # format
mypy                   # type-check
```

## Data

<!-- TODO: document datasets (Kepler KOI table, TESS TOI, Kaggle
     exoplanet archive...), download instructions, and licensing. -->

## Results

<!-- TODO: headline metrics table, key figures, comparison to
     published baselines (e.g. Astronet). -->

## Citation

<!-- TODO: BibTeX entry once the paper/preprint is available. -->

## License

MIT — see `LICENSE`.
