# Statistical Methods

This document defines the hypothesis tests and confidence intervals used in `exodet.benchmarking.statistics`.

## McNemar test

For paired binary classifiers A and B on the same labeled samples, define:

- \(n_{01}\): count where A is wrong and B is correct
- \(n_{10}\): count where A is correct and B is wrong

Under the null hypothesis of equal error rates, with continuity correction:

\[
\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}
\]

The p-value is \(P(\chi^2_1 > \chi^2)\). Significance at \(\alpha = 0.05\) is reported when \(p < 0.05\).

## Bootstrap confidence interval

For per-sample correctness indicators \(c_i \in \{0,1\}\), the point estimate is \(\bar{c}\). With \(B\) bootstrap replicates:

\[
\bar{c}^{(b)} = \frac{1}{n}\sum_i c_i^{(b)}, \quad c_i^{(b)} \sim \text{Uniform resample of } \{c_i\}
\]

The two-sided \((1-\alpha)\) percentile interval uses quantiles \(q_{\alpha/2}\) and \(q_{1-\alpha/2}\) of \(\{\bar{c}^{(b)}\}\).

## Paired Student *t*-test

Given paired scores \(a_i, b_i\), define differences \(d_i = a_i - b_i\). Test \(H_0: \mu_d = 0\) using the standard paired *t* statistic with \(n-1\) degrees of freedom. Used when comparing predicted probabilities on identical samples.

## Wilcoxon signed-rank test

Non-parametric alternative to the paired *t*-test. Ranks absolute differences \(|d_i|\) for \(d_i \neq 0\) and sums signed ranks. Appropriate when score differences are heavy-tailed or non-Gaussian.

## Calibration metrics

**Expected Calibration Error (ECE):**

\[
\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{n} \left| \text{acc}(b) - \text{conf}(b) \right|
\]

**Maximum Calibration Error (MCE):** \(\max_b |\text{acc}(b) - \text{conf}(b)|\) over bins with \(n_b > 0\).

**Brier score:** \(\frac{1}{n}\sum_i (p_i - y_i)^2\) for probabilities \(p_i\) and labels \(y_i\).

## Agreement interpretation

- McNemar: discriminative difference in **classification** decisions.
- Paired *t* / Wilcoxon: difference in **continuous scores** (e.g. probabilities).
- Bootstrap CI: uncertainty on **accuracy** without distributional assumptions.

All tests are two-sided unless noted. Multiple pairwise comparisons should be interpreted with appropriate multiplicity control in downstream publications.
