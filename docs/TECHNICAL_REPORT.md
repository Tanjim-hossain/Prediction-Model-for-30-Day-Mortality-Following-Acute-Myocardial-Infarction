# Technical Report

## Development and Internal Validation of a 30-Day Mortality Prediction Model after Acute Myocardial Infarction

### Executive summary

This project develops and internally validates a multivariable prediction model for 30-day mortality after acute myocardial infarction (AMI). The analysis uses 785 patients, including 52 deaths within 30 days (6.62%), and evaluates 17 candidate predictors.

The modelling strategy was designed around three constraints: the outcome is rare, the number of events is limited, and reliable probability prediction matters more than raw classification accuracy. For that reason, all imputation, scaling, categorical encoding, resampling and hyperparameter tuning were performed inside training folds, and performance was estimated with repeated nested stratified cross-validation.

The analysis evaluates penalised logistic regression, standard logistic regression, Random Forest, Gradient Boosting and an intercept-only reference. An initial Elastic Net search overwhelmingly selected the ridge/L2 endpoint. The final model is therefore ridge-penalised logistic regression with `C = 0.1` and no class rebalancing.

Internally validated performance was ROC-AUC 0.7881, PR-AUC 0.2783, Brier score 0.0556 and log loss 0.2070. Calibration was close to ideal, with intercept 0.1277, slope 1.0580 and mean predicted mortality risk of 6.63% compared with an observed event rate of 6.62%. Patient-level bootstrap resampling produced a ROC-AUC 95% interval of 0.7236–0.8478 and a PR-AUC interval of 0.1741–0.4054.

Class weighting and random oversampling did not improve the model; both substantially worsened Brier score and log loss. Sensitivity analyses supported treating Killip class as ordinal and retaining the simpler imputation-only approach. Age was the strongest predictive contributor in held-out permutation analysis, followed by Killip class, the heart-rate indicator, time to relief and weight.

The project is internally validated only. It does not establish causal effects, treatment benefit or transportability to new hospitals, countries or time periods.

---

## 1. Prediction objective

The target is a binary outcome:

`Day30_mortality = 1` if a patient died within 30 days and `0` otherwise.

The objective is to estimate an individual risk probability rather than merely produce a hard class label. This affects the entire evaluation strategy: discrimination metrics are reported, but proper probability scores and calibration are treated as first-class outcomes.

## 2. Dataset

The dataset contains:

- 785 patients;
- 52 deaths;
- 733 survivors;
- mortality prevalence 6.62%;
- 17 candidate predictors.

The predictor set includes demographic, clinical-history, physiological and event-related variables. The complete variable definitions are documented in `DATA_DICTIONARY.md`.

## 3. Data-quality audit

Before modelling, the notebook audits:

- schema and variable names;
- explicit missing values;
- encoded invalid or unknown values;
- duplicate rows;
- duplicate predictor profiles;
- clinically or numerically implausible values;
- outcome prevalence.

The raw audit identified seven explicitly missing source cells, three hypotension values stored as `Unknown`, 14 invalid `Killip_class = -1` values and two implausible height entries.

The following deterministic corrections were applied:

| Raw issue | Processing decision |
|---|---|
| `Hypothension` | rename to `Hypotension` |
| `Hyperthension` | rename to `Hypertension` |
| `Hypotension = Unknown` | recode as missing |
| `Killip_class = -1` | recode as missing |
| `Height = 1.75` | correct to 175 cm |
| `Height = 1690` | correct to 169 cm |

No patient was deleted. After recoding there were 24 missing predictor cells.

## 4. Exploratory analysis and modelling implications

The most important exploratory finding is the low event rate. With only 52 deaths, accuracy would be misleading because a classifier that predicts survival for nearly everyone could appear accurate.

The event count also creates a substantial overfitting risk relative to the number of candidate predictors. A single holdout test split would contain only a small number of deaths and would give unstable estimates.

These findings motivated:

- regularisation;
- repeated validation rather than one split;
- nested hyperparameter tuning;
- probability-focused evaluation;
- explicit calibration analysis;
- uncertainty quantification;
- conservative interpretation.

## 5. Predictor representation

Primary continuous or ordinal predictors:

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

- Smoking

Killip class is treated as ordinal in the primary analysis and categorical in a sensitivity analysis.

## 6. Leakage-safe preprocessing

All trainable preprocessing is performed inside the relevant model-training fold.

Continuous and ordinal predictors use:

- median imputation;
- standardisation.

Binary predictors use:

- most-frequent-value imputation.

Smoking uses:

- most-frequent-value imputation;
- one-hot encoding.

When random oversampling is evaluated, it is restricted to the training folds. This ensures that no validation-fold information contributes to imputation, scaling, category encoding, resampling or model selection.

## 7. Internal-validation architecture

The analysis uses repeated nested stratified cross-validation.

Outer validation:

- five stratified folds;
- five repeats;
- 25 held-out folds in total.

Inner tuning:

- four stratified folds inside each outer training sample;
- log loss as the hyperparameter-selection objective.

Every patient receives a genuine held-out probability in each outer repeat. The patient-level headline probability is the average of these repeated out-of-fold predictions. Repeat-specific metrics are retained to assess split-to-split stability.

## 8. Penalised logistic development

The initial model-development search uses Elastic Net logistic regression with:

- `C ∈ {0.01, 0.1, 1.0}`;
- `l1_ratio ∈ {0, 0.25, 0.5, 0.75, 1.0}`.

Twenty of the 25 outer folds selected `C = 0.1` and `l1_ratio = 0`, which corresponds to the ridge/L2 endpoint. The Elastic Net development model itself achieved approximately:

- ROC-AUC 0.7878;
- PR-AUC 0.2790;
- Brier score 0.0556;
- log loss 0.2071.

This result supported retaining all predictors with shrinkage instead of forcing a sparse L1 subset.

## 9. Candidate models

The common validation framework evaluates:

1. Elastic Net logistic regression;
2. ridge logistic regression;
3. standard logistic regression;
4. Random Forest;
5. Gradient Boosting;
6. intercept-only prevalence reference.

No univariable p-value screening is used to decide which predictors enter the final model.

## 10. Candidate-model performance

| Model | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| Ridge logistic | 0.7881 | 0.2783 | 0.0556 | 0.2070 |
| Random Forest | 0.7806 | 0.2462 | 0.0564 | 0.2094 |
| Standard logistic | 0.7726 | 0.2816 | 0.0558 | 0.2127 |
| Gradient Boosting | 0.7509 | 0.2379 | 0.0571 | 0.2160 |
| Intercept-only reference | 0.4638 | 0.0631 | 0.0619 | 0.2438 |

Ridge was selected because it offered the best overall balance of discrimination, probability error, calibration and repeated-validation stability. The final decision was not based on ROC-AUC alone.

## 11. Repeated-validation stability

The ridge model produced the following repeat-level means and standard deviations:

- ROC-AUC: 0.7831 ± 0.0056;
- PR-AUC: 0.2610 ± 0.0310;
- Brier score: 0.0559 ± 0.0009;
- log loss: 0.2084 ± 0.0026.

The comparatively small variation in ROC-AUC, Brier score and log loss supports the stability of the selected model under the repeated split design.

## 12. Class-imbalance ablation

Three matched ridge specifications were compared:

- no rebalancing;
- class weighting;
- random oversampling.

| Strategy | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| No rebalancing | 0.7881 | 0.2783 | 0.0556 | 0.2070 |
| Class weighting | 0.7746 | 0.2712 | 0.1711 | 0.5138 |
| Random oversampling | 0.7703 | 0.2700 | 0.1700 | 0.5108 |

Both rebalancing approaches substantially degraded probability quality. The final model therefore retains the natural class distribution.

## 13. Calibration

Calibration was evaluated from held-out predictions.

Selected ridge calibration:

- intercept 0.1277;
- slope 1.0580;
- mean predicted risk 0.0663;
- observed event rate 0.0662.

The slope is close to 1 and the average predicted risk is almost identical to the observed event rate.

The repository preserves the original calibration figure generated by the executed notebook.

## 14. Uncertainty estimation

Metric uncertainty was estimated using 2,000 patient-level bootstrap resamples of the final out-of-fold prediction set.

| Metric | Estimate | 95% bootstrap interval |
|---|---:|---:|
| ROC-AUC | 0.7881 | 0.7236–0.8478 |
| PR-AUC | 0.2783 | 0.1741–0.4054 |
| Brier score | 0.0556 | 0.0423–0.0690 |
| Log loss | 0.2070 | 0.1658–0.2487 |

These intervals quantify uncertainty in the preserved internally validated predictions and do not represent external-validation uncertainty.

## 15. Sensitivity analyses

Two key modelling assumptions were challenged.

### 15.1 Killip class representation

Treating Killip class categorically produced:

- ROC-AUC 0.7786;
- PR-AUC 0.2233;
- Brier 0.0567;
- log loss 0.2103.

The primary ordinal representation was retained.

### 15.2 Missingness indicators

Adding explicit missingness indicators produced:

- ROC-AUC 0.7889;
- PR-AUC 0.2787;
- Brier 0.0556;
- log loss 0.2069.

The improvement was negligible, so the simpler imputation-only strategy was retained.

## 16. Intercept-only reference

An intercept-only prevalence model was used as a baseline.

Its performance was:

- ROC-AUC 0.4638;
- PR-AUC 0.0631;
- Brier 0.0619;
- log loss 0.2438.

This provides a direct reference showing that the fitted models add predictive information beyond the marginal event rate.

## 17. Threshold-based operating characteristics

Four illustrative risk thresholds were evaluated.

| Threshold | Sensitivity | Specificity | PPV | NPV | Flagged |
|---|---:|---:|---:|---:|---:|
| 5% | 78.8% | 61.9% | 12.8% | 97.6% | 40.8% |
| 10% | 53.8% | 83.1% | 18.4% | 96.2% | 19.4% |
| 15% | 38.5% | 92.5% | 26.7% | 95.5% | 9.6% |
| 20% | 23.1% | 95.6% | 27.3% | 94.6% | 5.6% |

These thresholds are descriptive. They were not optimised as treatment cut-offs.

## 18. Decision-curve analysis

Decision-curve analysis compares the ridge model with treat-all and treat-none strategies.

Ridge net benefit:

- 5% threshold: 0.0335;
- 10%: 0.0181;
- 15%: 0.0131;
- 20%: 0.0051.

Across the evaluated grid, the ridge model had positive estimated net benefit over the relevant range relative to the reference strategies. This is not evidence that acting on the model causes improved patient outcomes.

## 19. Final ridge coefficients

The final model was refitted to all 785 patients using `C = 0.1`.

Largest absolute coefficients:

| Predictor | Coefficient | Odds ratio |
|---|---:|---:|
| Age | 0.6554 | 1.9258 |
| Heart-rate indicator | 0.4691 | 1.5985 |
| Time to relief >1 hour | 0.3208 | 1.3783 |
| Previous myocardial infarction | 0.3139 | 1.3687 |
| Killip class | 0.3072 | 1.3596 |
| Anterior infarct location | 0.2750 | 1.3165 |
| Gender | -0.2190 | 0.8033 |
| Weight | -0.2167 | 0.8052 |
| Diabetes | 0.2082 | 1.2315 |
| Hypotension | 0.1742 | 1.1903 |

Continuous and ordinal predictors are standardised, so their odds ratios correspond approximately to a one-standard-deviation increase.

## 20. Held-out permutation importance

Raw-predictor permutation importance was calculated in outer holdout folds. For each predictor:

1. the fold-specific ridge model remains fixed;
2. the raw predictor is permuted in the held-out fold;
3. performance is recalculated;
4. importance is measured primarily as the increase in held-out log loss.

Each predictor was permuted 20 times per outer split.

The strongest predictive contributors were:

1. Age;
2. Killip class;
3. Heart-rate indicator;
4. Time to relief;
5. Weight.

Permutation importance is predictive rather than causal and can be affected by correlated predictors.

## 21. Model packaging and deployment readiness

After model selection, the full-data ridge pipeline is fitted and serialized with `joblib`.

The executed notebook writes:

- `final_ridge_prediction_pipeline.joblib`;
- `final_ridge_model_metadata.json`.

The serialized pipeline includes preprocessing and the estimator together. This is important because future observations must receive the same transformation logic as the training data.

The repository demonstrates model packaging and inference readiness. It does not claim production API deployment, monitoring, clinical integration or external validation.

## 22. Reproducibility

The complete computational record is preserved in:

`notebooks/01_Complete_Executed_Analysis.ipynb`

The repository additionally contains:

- reusable Python source code;
- a separate Elastic Net screening module;
- machine-readable result tables;
- original analysis figures extracted from the executed notebook;
- tests and CI configuration;
- fixed random seed and environment metadata.

## 23. Limitations

The major limitations are:

- only 52 events;
- internal validation only;
- no independent external population;
- limited source-study metadata;
- possible predictor correlation;
- bootstrap intervals conditional on the preserved prediction procedure;
- illustrative threshold analysis;
- no causal interpretation;
- no production deployment or prospective monitoring.

## 24. Conclusion

Ridge-penalised logistic regression without class rebalancing provided the strongest overall internally validated performance among the evaluated models. The model combined useful discrimination with good probability calibration and lower probability error than the rebalanced alternatives. The analysis also demonstrates that class-imbalance corrections should be evaluated empirically rather than applied automatically.

The project provides a complete, auditable record from raw-data inspection through model packaging. Its main scientific limitation is the absence of external validation, which would be required before any real-world clinical use.
