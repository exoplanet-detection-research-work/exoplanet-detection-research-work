# Report generation

`exodet.reporting` produces per-candidate scientific reports after inference.

## CLI

```bash
exodet report -c configs/report.yaml
```

## Outputs (per candidate)

| Format | File |
|--------|------|
| JSON | `{sample_id}_report.json` |
| PDF | `{sample_id}_report.pdf` |
| CSV | `{sample_id}_summary.csv` |
| PNG | `{sample_id}_overview.png` |

## Report contents

Each report includes:

- Global and local phase-folded views
- Transit trapezoid fit overlay
- Classification probability, confidence, false-positive risk
- Refined transit parameter table
- Physical parameter table (when metadata available)
- Uncertainty intervals (when enabled)
- Links to explainability figures

## Configuration

```yaml
report:
  enabled: true
  output_dir: outputs/reports/reports
  formats: [json, pdf, csv]
  include_explainability: true
  include_transit_fit: true
  include_uncertainty: true
  top_n: 10                    # 0 = all above threshold
  probability_threshold: 0.1
  figure_dpi: 150
```

PDFs are generated with `matplotlib.backends.backend_pdf.PdfPages` (no extra dependencies).

## Programmatic use

```python
from exodet.reporting import ReportGenerator, run_report_generation

out_dir = run_report_generation("configs/report.yaml")
```

If inference has already been run, pass `inference_batch=` to avoid recomputation.
