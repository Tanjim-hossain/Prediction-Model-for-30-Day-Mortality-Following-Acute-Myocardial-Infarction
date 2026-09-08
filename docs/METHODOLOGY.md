# Methodology — Current ISDS Option A Analysis

## Objective

Develop and internally validate a multivariable model for predicting 30-day mortality after acute myocardial infarction in the supplied 785-patient dataset. The analysis follows the Option A assignment requirements and uses TRIPOD elements 8–20 as a reporting guide.

This is a **prediction-modelling** analysis. Predictor effects are therefore interpreted as predictive associations, not causal effects.

## 1. Data audit and cleaning

The raw dataset is preserved conceptually as the source record. Cleaning is deterministic and limited to issues supported by the assignment dictionary or obvious unit-entry errors:

- `Hypothension` → `Hypotension` (header correction).
- `Hyperthension` → `Hypertension` (header correction).
- `Hypotension = Unknown` → missing.
- `Killip_class = -1` → missing because valid classes are 1–4.
- `Height = 1.75` → 175 cm.
- `Height = 1690` → 169 cm.

No patient is deleted. After recoding there are 24 missing predictor cells and no exact duplicate rows.

## 2. Predictor representation

Primary continuous/ordinal block:

- Age
- Killip class
- Height
- Weight
- ST-elevation leads

Binary predictors:

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

Nominal categorical predictor:

- Smoking (three levels)

Killip class is treated as ordinal in the primary specification. A categorical representation is tested as a sensitivity analysis.

## 3. Leakage-safe preprocessing

Every transformation that can learn from the data is fitted inside the corresponding training fold:

- continuous/ordinal: median imputation + standardisation;
- binary: most-frequent-value imputation;
- smoking: most-frequent-value imputation + one-hot encoding;
- random oversampling, when tested: training folds only.

No global imputation, scaling, encoding or resampling is fitted before validation.

## 4. Validation design

The primary evaluation uses repeated nested cross-validation:

- outer loop: 5 stratified folds × 5 repeats = 25 held-out folds;
- inner loop: 4 stratified folds inside each outer training sample;
- hyperparameter selection criterion: log loss;
- preprocessing and model fitting are repeated from scratch inside each training split.

Each patient receives one genuine out-of-fold probability per repeat. The five probabilities are averaged for patient-level headline metrics. Repeat-specific metrics are retained to assess split-to-split stability.

## 5. Candidate models

The common internal-validation framework compares:

1. ridge-penalised logistic regression;
2. standard unpenalised logistic regression;
3. Random Forest;
4. Gradient Boosting;
5. intercept-only prevalence reference.

The original penalised search is elastic net with:

- `C ∈ {0.01, 0.1, 1}`
- `L1-ratio ∈ {0, 0.25, 0.5, 0.75, 1}`

Twenty of 25 outer folds select `C = 0.1` and `L1-ratio = 0`, supporting the ridge/L2 endpoint. The final penalised family is therefore implemented as ridge logistic regression with `lbfgs`.

No univariable p-value screen is used. Regularisation and held-out predictive performance control model complexity.

## 6. Class-imbalance ablation

Three otherwise matched ridge specifications are compared:

- no rebalancing;
- `class_weight="balanced"`;
- random oversampling.

The purpose is to test whether rebalancing improves *held-out probability prediction*, not to assume that rare outcomes automatically require resampling.

## 7. Performance measures

Discrimination:

- ROC-AUC
- PR-AUC

Probability accuracy:

- Brier score
- log loss

Calibration:

- calibration intercept (ideal 0)
- calibration slope (ideal 1)
- observed vs mean predicted event rate
- calibration curves

Rare-outcome accuracy is not used as a primary metric.

## 8. Uncertainty

For the final ridge model, 95% uncertainty intervals for ROC-AUC, PR-AUC, Brier score and log loss are estimated from 2,000 patient-level bootstrap resamples of the final internally validated out-of-fold predictions.

These intervals primarily quantify patient-sampling uncertainty conditional on the preserved internal-validation predictions; they do not capture every component of model-selection uncertainty.

## 9. Sensitivity analyses

Two prespecified sensitivity analyses are retained:

- Killip class encoded categorically instead of ordinally;
- explicit missingness indicators added to the primary preprocessing pipeline.

The simpler primary specification is retained unless the alternative meaningfully improves held-out performance.

## 10. Threshold analysis and decision curves

Illustrative risk thresholds of 5%, 10%, 15% and 20% are used to report:

- sensitivity;
- specificity;
- PPV;
- NPV;
- proportion classified high risk.

The thresholds are not optimized on the same data and are not treatment recommendations.

Decision-curve analysis evaluates net benefit over a 1–30% threshold range relative to treat-all and treat-none strategies.

## 11. Interpretation

The final ridge model is refitted to all 785 patients using the modal tuned `C = 0.1`.

- Coefficients and odds ratios are reported descriptively.
- Continuous/ordinal coefficients refer to standardized predictors.
- Raw-predictor permutation importance is evaluated in held-out outer folds; each predictor is shuffled repeatedly while the fold-specific model remains fixed.
- Importance is primarily the increase in held-out log loss after permutation.

Neither coefficients nor permutation importance establish causal effects.

## 12. Reproducibility

The current source implementation is `src/isds_option_a_pipeline.py`. It writes auditable tables, OOF predictions, tuning records, figures, final model metadata and a serialized full-data ridge pipeline to a user-specified output directory.

Key fixed settings:

- seed: 2026;
- outer CV: 5 folds × 5 repeats;
- inner CV: 4 folds;
- tuning metric: log loss;
- final ridge `C`: 0.1;
- class rebalancing: none;
- bootstrap repetitions: 2,000;
- permutation repetitions: 20 per predictor per outer split.
