# Model Card — AMI 30-Day Mortality Ridge Model

## Model details

- **Task:** binary risk prediction
- **Outcome:** death within 30 days after acute myocardial infarction
- **Model family:** ridge-penalised logistic regression
- **Penalty:** L2
- **Solver:** `lbfgs`
- **Final regularisation:** `C = 0.1`
- **Class rebalancing:** none
- **Development sample:** 785 patients, 52 deaths (6.62%)
- **Candidate predictors:** 17
- **Internal validation:** stratified 5-fold outer CV repeated 5 times
- **Hyperparameter tuning:** stratified 4-fold inner CV using log loss
- **Random seed:** 2026

## Intended use

Research and technical demonstration of probability-based prediction modelling, internal validation, calibration, uncertainty analysis and reproducible model packaging.

## Out-of-scope use

The model is not intended for direct diagnosis, triage, treatment allocation or clinical decision-making.

## Performance

| Metric | Estimate | 95% bootstrap interval |
|---|---:|---:|
| ROC-AUC | 0.7881 | 0.7236–0.8478 |
| PR-AUC | 0.2783 | 0.1741–0.4054 |
| Brier score | 0.0556 | 0.0423–0.0690 |
| Log loss | 0.2070 | 0.1658–0.2487 |

Calibration intercept = 0.1277; calibration slope = 1.0580; mean predicted risk = 6.63%; observed event rate = 6.62%.

## Model-selection rationale

The Elastic Net development search selected `C = 0.1, l1_ratio = 0` in 20 of 25 outer folds. Ridge then provided the best overall balance of discrimination, proper probability scoring, calibration and repeated-validation stability.

## Known limitations

- 52 events only;
- internal validation only;
- no temporal, geographic or external-site validation;
- limited source-study metadata;
- predictor correlation can affect coefficient and permutation-importance interpretation;
- thresholds are descriptive;
- no causal effect or treatment-benefit claims.

## Packaging

The complete notebook serializes the fitted preprocessing-plus-model pipeline with `joblib` and writes accompanying model metadata. The repository intentionally documents this as **model packaging / deployment readiness**, not production deployment.

## Interpretation warning

Coefficients, odds ratios and permutation importance are predictive associations, not causal effects.
