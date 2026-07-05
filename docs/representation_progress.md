# Representation layer — implementation progress

Working notes for the ML representation stage (`exodet.representation`).

## Status

- [x] Progress file
- [x] containers.py — PhaseFoldedCurve, View, FeatureVector, DatasetSample, RepresentationDataset
- [x] folding.py — phase fold + transit alignment (Module 1 & 4)
- [x] views.py — global (2001) + local (401) views, vectorized binning (Module 2 & 3)
- [x] features.py — physics feature extractor (Module 5)
- [x] scaling.py — standard/robust/minmax + log + inverse (Module 6)
- [x] splitting.py — star/candidate/stratified/grouped + leakage guard (Module 8)
- [x] cache.py — SHA-256 NPZ cache + mmap (Module 9)
- [x] augmentation.py — 5 physically valid augmenters (Module 10)
- [x] config.py + pipeline.py + runner.py + CLI `dataset` + example YAML
- [x] visualization/representation.py (Module 11)
- [x] tests — 51 unit + integration + perf (`test_representation*.py`)
- [x] benchmark script + full run 1..10000 (Module 12)

## Verified results

- Full suite: **249 passed** (+51 representation tests).
- Benchmark (201/81 bins, 4k–10k cadences, macOS arm64):
  - 1 → 0.001 s/sample, 10k → 8.77 s total (~0.001 s/sample)
  - Scaling factor ~1.0 (linear); RSS 160 MB → 1.3 GB at 10k
  - Disk ~3.4 KB/sample (NPZ+JSON sidecar)
  - Cache: 20 warm reads on 100 samples ≪ 1 s
- `exodet dataset -c configs/representation_example.yaml` end-to-end OK.

## Key design decisions

- Phase: φ = ((t − epoch)/P + 0.5) mod 1 − 0.5 → transit at 0.
- Alignment: flux-weighted dip centroid within ±1 duration; cap ±0.5 duration.
- Binning: lexsort + bincount exact medians (no Python loops).
- View normalization: `none` | `center` | `astronet` (median→0, min→−1).
- Scaler fitted on **train split only** (runner); cache keyed by SHA-256 fingerprint.
- Star-level split default; candidate/stratified require `allow_star_leakage=true`.

## Usage

```bash
exodet dataset -c configs/representation_example.yaml
python scripts/benchmark_representation.py --sizes 1 10 100 1000 10000
```

Outputs: `data/processed/dataset/{train,validation,test}.npz`, scaler JSON,
CSV summaries, diagnostic figures in `outputs/figures/`.
