# Methodology

## Objective

Develop and internally validate a multivariable model for predicting 30-day mortality after acute myocardial infarction.

This is a prediction-modelling analysis. Predictor coefficients and importance measures are interpreted as predictive associations, not causal effects.

## Data audit and deterministic cleaning

The analysis preserves the source data and applies only explicit, auditable corrections:

- `Hypothension` → `Hypotension`
- `Hyperthension` → `Hypertension`
- `Hypotension = Unknown` → missing
- `Killip_class = -1` → missing
- `Height = 1.75` → 175 cm
- `Height = 1690` → 169 cm

No patient is deleted. After recoding there are 24 missing predictor cells.

## Predictor representation

Continuous / ordinal:

- Age
- Killip class
- Height
- Weight
- ST-elevation leads

Binary:

- Gender
- Diabetes
- Hypotension
- Heart-rate indicator
- Anterior infarct location
- Previous myocardial infarction
- Hypertension
- Hypercholesterolaemia
- Previous angina pectoris
- Family history of MI
- Time to relief >1 hour

Categorical:

- Smoking

Killip class is ordinal in the primary analysis and categorical in a sensitivity analysis.

## Leakage-safe preprocessing

All learned transformations are estimated within training folds:

- continuous/ordinal: median imputation + standardisation;
- binary: most-frequent-value imputation;
- smoking: most-frequent-value imputation + one-hot encoding;
- oversampling: training folds only.

## Validation design

- outer loop: 5 stratified folds × 5 repeats;
- inner loop: 4 stratified folds;
- tuning criterion: log loss;
- seed: 2026.

Preprocessing, resampling, tuning and fitting are repeated from scratch inside each training split.

## Model development

The full analysis evaluates:

- Elastic Net logistic regression;
- ridge logistic regression;
- standard logistic regression;
- Random Forest;
- Gradient Boosting;
- intercept-only reference.

The Elastic Net grid is:

- `C ∈ {0.01, 0.1, 1}`;
- `l1_ratio ∈ {0, 0.25, 0.5, 0.75, 1}`.

Twenty of 25 outer folds select `C = 0.1` and `l1_ratio = 0`, supporting the ridge endpoint.

## Class-imbalance experiments

Matched ridge variants compare:

- no rebalancing;
- balanced class weights;
- random oversampling.

The choice is based on held-out predictive performance.

## Performance metrics

Discrimination:

- ROC-AUC
- PR-AUC

Probability quality:

- Brier score
- log loss

Calibration:

- intercept
- slope
- observed versus mean predicted risk
- calibration curves

## Uncertainty

The final held-out ridge predictions are resampled 2,000 times at patient level to estimate 95% bootstrap intervals for ROC-AUC, PR-AUC, Brier score and log loss.

## Sensitivity analyses

- ordinal versus categorical Killip class;
- imputation only versus explicit missingness indicators.

## Threshold and decision-curve analysis

Illustrative thresholds of 5%, 10%, 15% and 20% are used to report sensitivity, specificity, PPV, NPV and proportion flagged.

Decision-curve net benefit is evaluated over a 1%–30% threshold range.

## Interpretation

The final ridge model is refitted on all available patients with `C = 0.1`.

Interpretation uses:

- shrunk coefficients;
- odds ratios;
- outer-fold raw-predictor permutation importance.

Permutation importance is calculated in held-out folds with 20 shuffles per predictor per outer split.

## Model packaging

The final preprocessing-plus-model pipeline is serialized with `joblib`, and model metadata are exported to JSON.

## Full computational record

See `../notebooks/01_Complete_Executed_Analysis.ipynb` for all executed code, outputs, figures, assertions and model-packaging steps.
