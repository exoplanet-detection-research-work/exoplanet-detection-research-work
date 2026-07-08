# Cross-Platform Compatibility Report

ExoDet supports **Windows 10/11**, **macOS (Intel + Apple Silicon)**, and **Linux (Ubuntu)** from a single codebase without manual code edits.

## Summary

| Area | Status |
|------|--------|
| Unix-only `resource` module | **Removed** — replaced with `psutil` |
| macOS-only `sysctl` memory probe | **Removed** — replaced with `psutil.virtual_memory()` |
| Hardcoded `/tmp` paths | **Removed** — `tempfile.gettempdir()` / pytest `tmp_path` |
| Windows reserved filenames | **Mitigated** — `safe_filename()` on report paths |
| `os.path` / `os.fork` / `SIGUSR` | **Not used** in source |
| `subprocess` with `shell=True` | **Not used** |
| Symlinks | **Utility added** — `link_or_copy()` with copy fallback |

## Unix-Specific Dependencies Found and Replaced

### 1. `resource` module (POSIX-only)

**Crash on Windows:** `ModuleNotFoundError: No module named 'resource'`

| File | Before | After |
|------|--------|-------|
| `src/exodet/benchmarking/runner.py` | `resource.getrusage(RUSAGE_SELF).ru_maxrss` | `process_rss_bytes()` via `psutil` |
| `src/exodet/experiments/performance.py` | `resource.getrusage(...)` | `process_rss_bytes()` |
| `scripts/benchmark_representation.py` | `resource.getrusage(...)` | `process_rss_bytes()` |
| `scripts/benchmark_tce.py` | `resource.getrusage(...)` | `process_rss_bytes()` |

**New module:** `src/exodet/utils/process_metrics.py`

Provides portable:
- `process_rss_bytes()` — resident memory (bytes)
- `process_cpu_seconds()` — user CPU time
- `process_num_threads()` — thread count
- `system_memory_bytes()` — total RAM
- `process_stats()` — combined snapshot

**Dependency added:** `psutil>=5.9` in `pyproject.toml` and `requirements.txt`

> Note: `ru_maxrss` units differ between Linux (KB) and macOS (bytes). The old scripts divided by `1e6` assuming bytes, which was incorrect on Linux. `psutil` returns consistent byte values on all platforms.

### 2. `sysctl` subprocess (macOS-only)

| File | Before | After |
|------|--------|-------|
| `src/exodet/reproducibility/collector.py` | `subprocess.run(["sysctl", "-n", "hw.memsize"])` | `system_memory_bytes()` via `psutil` |

`git rev-parse` in `ml/tracking.py` remains — `git` is optional and failures are caught.

### 3. Hardcoded `/tmp` paths

| File | Before | After |
|------|--------|-------|
| `src/exodet/experiments/performance.py` | `f"/tmp/ckpt_{i}.pt"` | `Path(tempfile.gettempdir()) / "exodet_bench" / f"ckpt_{i}.pt"` |
| `tests/test_representation.py` | `RepresentationCache("/tmp/x", ...)` | `Path(tempfile.gettempdir()) / "exodet_test_cache"` |

### 4. Windows reserved filenames

| File | Change |
|------|--------|
| `src/exodet/utils/paths.py` | **New** — `safe_filename()`, `link_or_copy()` |
| `src/exodet/inference/pipeline.py` | Explainability output prefixes sanitized |
| `src/exodet/reporting/report.py` | Report file prefixes sanitized |
| `src/exodet/reporting/runner.py` | Report directory names sanitized |
| `src/exodet/visualization/representation.py` | Figure slug sanitized |

### 5. Documentation (Unix-only shell commands)

| File | Change |
|------|--------|
| `README.md` | Windows + macOS/Linux venv activation |
| `docs/CO_RESEARCHER_SETUP.txt` | Cross-platform install, cleanup, cache clear |
| `docs/EXODET_RESEARCH_GUIDE.txt` | Cross-platform setup commands |

## Files Modified

### Source code
- `src/exodet/benchmarking/runner.py`
- `src/exodet/experiments/performance.py`
- `src/exodet/reproducibility/collector.py`
- `src/exodet/inference/pipeline.py`
- `src/exodet/reporting/report.py`
- `src/exodet/reporting/runner.py`
- `src/exodet/visualization/representation.py`
- `src/exodet/utils/process_metrics.py` *(new)*
- `src/exodet/utils/paths.py` *(new)*

### Scripts
- `scripts/benchmark_representation.py`
- `scripts/benchmark_tce.py`

### Tests
- `tests/test_representation.py`
- `tests/test_process_metrics.py` *(new)*
- `tests/test_paths.py` *(new)*

### Packaging & docs
- `pyproject.toml`
- `requirements.txt`
- `README.md`
- `docs/CO_RESEARCHER_SETUP.txt`
- `docs/EXODET_RESEARCH_GUIDE.txt`
- `docs/CROSS_PLATFORM_COMPATIBILITY.md` *(this file)*

## Audited — No Changes Required

| Pattern | Result |
|---------|--------|
| `os.uname()` | Not found |
| `os.fork()` | Not found |
| `signal.SIGUSR1/2` | Not found |
| `os.sched_getaffinity()` | Not found |
| `chmod` / `stat.S_*` | Not found |
| `os.symlink()` in use | Not found (utility added for future use) |
| `os.path.*` in `src/` | Not found (already uses `pathlib`) |
| `subprocess` + `shell=True` | Not found |
| `multiprocessing` fork | Not found |

`tracemalloc` is used in several benchmark modules — it is **cross-platform** (stdlib) and was left unchanged.

## Remaining Platform-Specific Limitations

1. **`git` CLI** — optional for reproducibility metadata; silently skipped if not installed.
2. **PyTorch MPS** — Apple Silicon GPU only; CUDA on NVIDIA Linux/Windows; CPU fallback everywhere.
3. **Lightkurve cache paths** — OS-specific user cache dirs (`~/.cache` vs `%USERPROFILE%\.cache`); documented in setup guide.
4. **Long path names on Windows** — very deep output trees may hit the 260-character legacy limit unless long-path support is enabled in Windows.
5. **Attention explainability** — known fusion-model bug (unrelated to OS); fails on all platforms when enabled.

## Scientific Behavior Preserved

No changes were made to:
- ML models, training, or inference logic
- Dataset formats or generation
- TCE search, representations, or evaluation metrics
- Scientific calculations or report content

Only portability infrastructure changed.

## Verification

```bash
pip install -e ".[dev]"
pytest -q
exodet info
exodet benchmark -c configs/benchmark_example.yaml  # if dataset present
```

All CLI entry points import without POSIX-only modules on any platform.
