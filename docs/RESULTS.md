# Current Internally Validated Results

These are the **current** ISDS Option A results. They replace the older January-2026 weighted-logistic results still preserved in legacy root artifacts.

## Dataset

- Patients: **785**
- Deaths within 30 days: **52**
- Event rate: **6.62%**
- Candidate predictors: **17**
- Missing predictor cells after auditable recoding: **24**
- Exact duplicate rows: **0**

## Candidate-model comparison

| Model | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| **Ridge logistic** | **0.7881** | 0.2783 | **0.0556** | **0.2070** |
| Random Forest | 0.7806 | 0.2462 | 0.0564 | 0.2094 |
| Standard logistic | 0.7726 | **0.2816** | 0.0558 | 0.2127 |
| Gradient Boosting | 0.7509 | 0.2379 | 0.0571 | 0.2160 |
| Intercept-only reference | 0.4638 | 0.0631 | 0.0619 | 0.2438 |

Ridge was retained because it provided the best overall balance of discrimination, proper scoring-rule performance, calibration and repeated-split stability. Model selection was not based on ROC-AUC alone.

## Repeat-wise ridge stability

Mean repeat-level estimates reported in the current analysis:

- ROC-AUC: **0.7831**, SD **0.0056**
- PR-AUC: **0.2610**, SD **0.0310**
- Brier: **0.0559**, SD **0.0009**
- Log loss: **0.2084**, SD **0.0026**

## Class-imbalance ablation

| Strategy | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| **No rebalancing** | **0.7881** | **0.2783** | **0.0556** | **0.2070** |
| Class weighting | 0.7746 | 0.2712 | 0.1711 | 0.5138 |
| Random oversampling | 0.7703 | 0.2700 | 0.1700 | 0.5108 |

Both rebalancing strategies substantially worsened Brier score and log loss. The final model therefore keeps the natural event prevalence.

## Calibration

Selected ridge model:

- calibration intercept: **0.1277**
- calibration slope: **1.0580**
- mean predicted mortality risk: **6.63%**
- observed mortality: **6.62%**

## Bootstrap uncertainty

2,000 patient-level bootstrap resamples of the final OOF prediction set:

| Metric | Estimate | 2.5th percentile | 97.5th percentile |
|---|---:|---:|---:|
| ROC-AUC | 0.7881 | 0.7236 | 0.8478 |
| PR-AUC | 0.2783 | 0.1741 | 0.4054 |
| Brier | 0.0556 | 0.0423 | 0.0690 |
| Log loss | 0.2070 | 0.1658 | 0.2487 |

## Sensitivity analyses

| Specification | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| Primary ridge | 0.7881 | 0.2783 | 0.0556 | 0.2070 |
| Killip categorical | 0.7786 | 0.2233 | 0.0567 | 0.2103 |
| Missing indicators added | 0.7889 | 0.2787 | 0.0556 | 0.2069 |

The ordinal Killip representation is retained. Missing indicators yield negligible improvement, so the simpler imputation-only strategy is retained.

## Threshold trade-offs

| Threshold | Sensitivity | Specificity | PPV | NPV | Flagged high risk |
|---|---:|---:|---:|---:|---:|
| 5% | 78.8% | 61.9% | 12.8% | 97.6% | 40.8% |
| 10% | 53.8% | 83.1% | 18.4% | 96.2% | 19.4% |
| 15% | 38.5% | 92.5% | 26.7% | 95.5% | 9.6% |
| 20% | 23.1% | 95.6% | 27.3% | 94.6% | 5.6% |

These thresholds are illustrative and are not recommended clinical action thresholds.

## Decision-curve analysis

Ridge net benefit at selected thresholds:

- 5%: **0.0335**
- 10%: **0.0181**
- 15%: **0.0131**
- 20%: **0.0051**

The internally validated decision curve is positive over the evaluated threshold range relative to the reference strategies, but this does not establish causal benefit from any intervention.

## Final ridge coefficients

The final full-data ridge fit uses `C = 0.1`. Largest coefficients by absolute magnitude:

| Predictor | Coefficient | Odds ratio |
|---|---:|---:|
| Age | 0.6554 | 1.9258 |
| Heart-rate indicator | 0.4691 | 1.5985 |
| Time to relief >1 hour | 0.3208 | 1.3783 |
| Previous myocardial infarction | 0.3139 | 1.3687 |
| Killip class | 0.3072 | 1.3596 |
| Anterior infarct location | 0.2750 | 1.3165 |
| Gender (female) | -0.2190 | 0.8033 |
| Weight | -0.2167 | 0.8052 |
| Diabetes | 0.2082 | 1.2315 |
| Hypotension | 0.1742 | 1.1903 |

Continuous/ordinal variables are standardized before modelling, so their odds ratios refer approximately to a one-standard-deviation increase.

## Held-out permutation importance

Outer-fold raw-predictor permutation importance identified the strongest contributors in this order:

1. Age
2. Killip class
3. Heart-rate indicator
4. Time to relief
5. Weight

Small negative importance values for weak predictors can occur because of sampling variability and predictor correlation and should not be interpreted as protective effects.

## Interpretation boundary

All model coefficients and importance values describe **predictive association** in this dataset. They do not establish causal effects, biological mechanisms, or treatment benefit. External validation is required before clinical use.
