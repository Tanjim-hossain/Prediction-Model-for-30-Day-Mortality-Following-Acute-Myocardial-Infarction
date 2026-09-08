# Model card — AMI 30-day mortality ridge model

## Model details

- **Task:** binary risk prediction
- **Outcome:** death within 30 days after acute myocardial infarction
- **Model family:** ridge-penalised logistic regression
- **Penalty:** L2
- **Solver:** `lbfgs`
- **Final regularisation strength:** `C = 0.1`
- **Class rebalancing:** none
- **Development sample:** 785 patients, 52 deaths (6.62%)
- **Candidate predictors:** 17
- **Internal validation:** stratified 5-fold outer CV repeated 5 times
- **Hyperparameter tuning:** stratified 4-fold inner CV using log loss
- **Random seed:** 2026

## Intended use

Academic demonstration of modern prediction-model development, internal validation and transparent reporting.

## Out-of-scope use

The model is **not** intended for direct clinical decision-making, triage, diagnosis, treatment allocation or deployment in a healthcare system.

## Performance

| Metric | Estimate | 95% bootstrap CI |
|---|---:|---:|
| ROC-AUC | 0.7881 | 0.7236–0.8478 |
| PR-AUC | 0.2783 | 0.1741–0.4054 |
| Brier score | 0.0556 | 0.0423–0.0690 |
| Log loss | 0.2070 | 0.1658–0.2487 |

Calibration intercept = 0.1277; calibration slope = 1.0580; mean predicted risk = 6.63%; observed event rate = 6.62%.

## Model-selection rationale

An elastic-net development search overwhelmingly selected the L2/ridge end of the path: `C=0.1, l1_ratio=0` in 20 of 25 outer folds. Ridge then gave the best overall balance of discrimination, probability scoring, calibration and repeated-CV stability among the candidate models.

## Known limitations

- Only 52 events were available.
- All validation is internal.
- No temporal, geographic or external-site validation was available.
- The dataset source materials do not establish transportability to contemporary clinical settings.
- Predictors may be correlated, so coefficient magnitude and permutation importance are not independent measures of clinical importance.
- Threshold and decision-curve analyses are exploratory and do not define a treatment policy.

## Interpretation warning

Coefficients, odds ratios and permutation importance are **predictive associations**, not causal effects.
