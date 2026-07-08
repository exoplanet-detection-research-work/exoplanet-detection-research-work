# EXODET — CO-RESEARCHER SETUP & ARTIFACT GUIDE (Git Bash on Windows)

Read this **before** running any commands. It prevents the most common confusion:
"training succeeded in 0.06s but `outputs/` looks empty."

Good news: Git Bash is a real bash shell, so almost every command from your co-researcher's macOS guide works **completely unchanged** — `rm -f`, `rm -rf`, `cp`, `ls -la`, backslash line continuations, and multi-line `python -c "..."` all behave identically. The only real differences are a couple of Windows filesystem quirks around the virtual environment, flagged inline below.

---

## 1. What is in Git vs. what you generate locally

**COMMITTED** (cloning gives you these):
- All Python source: `src/exodet/`
- All configs: `configs/`
- All tests: `tests/`
- TIC target catalogs: `data/tic_training_catalog.csv`, `data/tics_*.csv`
- Documentation: `docs/EXODET_RESEARCH_GUIDE.txt` (full science + ML guide)
- Scripts: `scripts/`

**NOT IN GIT** (you must generate these by running the pipeline):
- `data/raw/*.npz` — downloaded TESS light curves
- `data/processed/*.npz` — preprocessed curves
- `data/processed/dataset/` — `train.npz`, `validation.npz`, `test.npz`
- `outputs/checkpoints/` — model weights (`.pt` or `.joblib`)
- `outputs/reports/` — TCE catalogs, evaluation JSON
- `outputs/logs/` — per-run log files
- `outputs/figures/` — diagnostic plots

This is intentional: binary data and run outputs are large and machine-specific.

---

## 2. The #1 artifact confusion (shadow test)

**SYMPTOM:**
```
exodet train
```
prints "Training complete" in ~0.06s, then
```
ls outputs/checkpoints/
```
looks empty (only `.gitkeep`) → you conclude "training is fake / no artifacts."

**REALITY:**
Artifacts are in a **subdirectory named after the experiment**:

sklearn backend (`logistic_regression`):
```
outputs/checkpoints/exodet_incremental_v1/model.joblib
outputs/checkpoints/exodet_incremental_v1/model.json
outputs/checkpoints/exodet_incremental_v1/training_report.json
```

PyTorch fusion backend (`training_research.yaml`):
```
outputs/checkpoints/fusion_research_v1/last.pt
outputs/checkpoints/fusion_research_v1/best.pt
outputs/checkpoints/fusion_research_v1/ema.pt
outputs/checkpoints/fusion_research_v1/swa.pt
outputs/checkpoints/fusion_research_v1/training_report.json
outputs/checkpoints/fusion_research_v1/metrics.csv
outputs/checkpoints/fusion_research_v1/tensorboard/
```

The CLI prints the **parent** directory (`outputs/checkpoints`), not the file path.

**VERIFY AFTER TRAINING:**
```bash
ls -la outputs/checkpoints/<experiment_name>/
```

---

## 3. Installation

```bash
git clone https://github.com/exoplanet-detection-research-work/exoplanet-detection-research-work.git
cd exoplanet-detection-research-work
python -m venv .venv
source .venv/Scripts/activate
pip install -e ".[dev]"

# Required for real TESS downloads:
pip install lightkurve astropy

# Verify install:
exodet info
pytest -q
```

**The two things that actually differ from macOS here:**

- **`python` not `python3`**: the standard python.org Windows installer only puts `python` on PATH, not `python3`. Use `python`. (If you installed Python via MSYS2's `pacman`, `python3` will also work — either is fine, just be consistent for the rest of this guide.)
- **`Scripts/` not `bin/`**: on Windows, `venv` creates `.venv/Scripts/` instead of `.venv/bin/`. It still generates a plain bash-compatible activation script there (no `.ps1` or `.bat` extension) specifically for Git Bash/MSYS shells, so `source .venv/Scripts/activate` is the correct equivalent of `source .venv/bin/activate`.

After activation your prompt should show `(.venv)`, and `which exodet` should point inside `.venv/Scripts/`.

---

## 4. Reproduce the real-data E2E run (from committed CSV catalogs)

### STEP 0 — Clear any stale local state from prior experiments

```bash
rm -f data/processed/dataset/train.npz data/processed/dataset/train.json
rm -f data/processed/dataset/validation.npz data/processed/dataset/test.npz
rm -f data/processed/dataset/manifest.json data/processed/dataset/feature_scaler.json
rm -f data/processed/dataset_registry.json
rm -rf data/interim/update_state

# If lightkurve downloads fail with corrupt cache errors:
rm -rf ~/.cache/lightkurve/mastDownload
```

This block is identical to the macOS version — `~` resolves via Git Bash's `$HOME`, which Git for Windows sets to your Windows user profile by default, so it lands in the right place without any changes.

> If that path doesn't exist and the corrupt-cache error persists, `lightkurve`'s cache location has moved between versions on some setups — check `~/.lightkurve-cache/mastDownload` as a fallback (run `find ~ -maxdepth 1 -iname "*lightkurve*"` to see which one your install actually uses).

### STEP 1 — Ingest POSITIVE TICs (label=1)

```bash
exodet update -c configs/update.yaml \
  --tic-file data/tics_positive.csv \
  -o labeling.default_label=1 \
  -o update.resume_training=false \
  -o update.evaluation.enabled=false \
  -o update.force_reprocess=true
```

Expect: 30–60+ minutes (real MAST downloads + BLS on long baselines).

### STEP 2 — Ingest NEGATIVE TICs (label=0)

```bash
exodet update -c configs/update.yaml \
  --tic-file data/tics_negative.csv \
  -o labeling.default_label=0 \
  -o update.resume_training=false \
  -o update.evaluation.enabled=false
```

Note: TIC 400799224 has no SPOC TESS data on MAST (expected failure).

### STEP 3 — Verify dataset

```bash
python -c "
from exodet.representation.containers import RepresentationDataset
d = RepresentationDataset.load('data/processed/dataset/train.npz')
print('samples:', len(d), 'labels:', sorted(s.label for s in d.samples))
"
```

Expect: ~40 samples, labels `[0, 1]` mixed.

### STEP 4 — Train fusion model

```bash
exodet train -c configs/training_research.yaml \
  -o training.epochs=15 \
  -o training.batch_size=4 \
  -o training.trainer.params.research.augmentation.enabled=false \
  -o training.trainer.params.research.imbalance.enabled=false \
  -o training.trainer.params.research.curriculum.enabled=false
```

Verify: `ls outputs/checkpoints/fusion_research_v1/`

### STEP 5 — Evaluate

```bash
cp outputs/checkpoints/fusion_research_v1/last.pt \
   outputs/checkpoints/fusion_research_v1/best.pt
exodet evaluate -c configs/training_research.yaml
```

### STEP 6 — Inference + reports

```bash
exodet infer -c configs/report.yaml -o inference.explainability.enabled=false
exodet report -c configs/report.yaml \
  -o inference.explainability.enabled=false \
  -o report.include_explainability=false
```

---

## 5. Fast sanity check (sklearn, seconds not hours)

After Step 1+2 above, or with any labeled dataset in `train.npz`:

```bash
exodet train -c configs/update.yaml \
  -o model.architecture.name=logistic_regression \
  -o training.trainer.params.backend=sklearn \
  -o training.trainer.params.use_views=features_only
```

Artifacts: `outputs/checkpoints/exodet_incremental_v1/model.joblib`
**NOT** `best.pt` (that's PyTorch-only).

---

## 6. Key config files (what each is for)

| File | Purpose |
|---|---|
| `configs/update.yaml` | full incremental ingest (download → TCE → dataset) |
| `configs/tce_example.yaml` | standalone BLS search |
| `configs/representation_example.yaml` | standalone dataset build |
| `configs/training_research.yaml` | fusion CNN+Transformer training |
| `configs/report.yaml` | inference + scientific reports |

---

## 7. Labeling rule

The CSV label column is **not** auto-read. Use two update runs:
- `data/tics_positive.csv` with `-o labeling.default_label=1`
- `data/tics_negative.csv` with `-o labeling.default_label=0`

See `data/tic_training_catalog.csv` for the full target list and science notes.

---

## 8. Where to read more

- `docs/EXODET_RESEARCH_GUIDE.txt` — complete pipeline + model math (1000+ lines)
- `docs/incremental_training.md` — update pipeline API
- `docs/dataset_versioning.md` — dataset registry semantics

---

## A few Windows-only things worth knowing about

- **`exodet: command not found` after activation** — this almost always means `.venv/Scripts` didn't make it onto PATH. Run `which exodet` to check; if empty, re-run `source .venv/Scripts/activate` and confirm your prompt shows `(.venv)`.
- **MSYS path conversion** — Git Bash auto-rewrites arguments that look like absolute POSIX paths (a leading `/`) into Windows paths, which can occasionally mangle an unrelated flag. None of the commands above start with a leading `/`, so it shouldn't bite here — but if you add your own commands and see a path get silently rewritten, prefix it with an extra `/` or run `export MSYS_NO_PATHCONV=1` first.
- **Antivirus/Defender** can noticeably slow the MAST downloads and BLS search in Steps 1–2 since it scans each new file as it lands. Worth adding the repo folder to Defender exclusions if downloads feel unusually slow.
- **Path length limits**: deeply nested cache paths (`mastDownload/...`) can occasionally hit Windows' historical 260-character path limit. If you get mysterious "file not found" errors mid-download, run `git config --system core.longpaths true` or move the repo closer to your drive root (e.g. `C:/dev/exodet`).
