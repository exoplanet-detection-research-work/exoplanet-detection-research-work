# Explainability

`ExplainabilityEngine` generates publication-quality figures for trained hybrid models.

## Methods

| Method | Description |
|--------|-------------|
| **Grad-CAM** | Gradient-weighted class activation on the local CNN encoder |
| **Integrated Gradients** | Path integral from baseline to input along view bins |
| **Attention heatmap** | CLS attention weights from the global transformer |
| **Attention rollout** | Same attention map (CLS-averaged) |
| **Occlusion sensitivity** | Mask contiguous phase bins; measure \(\|\Delta P\|\) |
| **Feature attribution** | Input gradients on physics feature vector |

## Grad-CAM

For CNN activation map \(A\) and gradient \(G\):

\[
\alpha_k = \frac{1}{L}\sum_l \frac{\partial y}{\partial A_{k,l}}, \quad
\mathrm{CAM} = \mathrm{ReLU}\left(\sum_k \alpha_k A_k\right)
\]

## Integrated Gradients

\[
\mathrm{IG}_i(x) = (x_i - x'_i) \int_{\alpha=0}^{1} \frac{\partial f(x' + \alpha(x-x'))}{\partial x_i}\, d\alpha
\]

approximated with a Riemann sum over `n_integrated_steps`.

## Configuration

```yaml
inference:
  explainability:
    enabled: true
    methods:
      - grad_cam
      - integrated_gradients
      - attention
      - occlusion
      - feature_attribution
    dpi: 150
    n_integrated_steps: 32
```

Figures are written under `{figure_dir}/explainability/` and referenced in candidate reports and catalogs.

## Uncertainty (related)

Classification uncertainty (MC dropout, bootstrap) is configured separately:

```yaml
inference:
  uncertainty:
    method: mc_dropout   # none | mc_dropout | bootstrap
    n_samples: 30
    credible_alpha: 0.68
```

Credible intervals use symmetric quantiles at \((1-\alpha)/2\) and \(1-(1-\alpha)/2\).
