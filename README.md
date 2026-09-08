# 30-Day Mortality Prediction after Acute Myocardial Infarction

**Inference for Statistics and Data Science — Option A (Prediction Modelling)**  
**Hasselt University | MSc Statistics and Data Science**

This repository documents an end-to-end prediction-modelling workflow for estimating **30-day mortality risk after acute myocardial infarction (AMI)** from the course-provided dataset of **785 patients**, including **52 deaths (6.62%)** and **17 candidate predictors**.

The current analysis is a **prediction** project, not a causal analysis. All headline results below come from leakage-safe **internal validation**. The model has **not** been externally validated and is **not intended for clinical deployment**.

## Current headline result

The selected model is **ridge-penalised logistic regression** with `C = 0.1` and **no class rebalancing**.

| Metric | Internally validated estimate | 95% bootstrap CI |
|---|---:|---:|
| ROC-AUC | **0.7881** | 0.7236–0.8478 |
| PR-AUC | **0.2783** | 0.1741–0.4054 |
| Brier score | **0.0556** | 0.0423–0.0690 |
| Log loss | **0.2070** | 0.1658–0.2487 |

Calibration was close to ideal: **intercept 0.1277**, **slope 1.0580**, with mean predicted mortality risk **6.63%** versus observed mortality **6.62%**.

## Why this model was selected

The project did not select a model from a single train/test split or from ROC-AUC alone. Candidate methods were evaluated under the same repeated nested cross-validation framework.

| Model | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| **Ridge logistic** | **0.7881** | 0.2783 | **0.0556** | **0.2070** |
| Random Forest | 0.7806 | 0.2462 | 0.0564 | 0.2094 |
| Standard logistic | 0.7726 | **0.2816** | 0.0558 | 0.2127 |
| Gradient Boosting | 0.7509 | 0.2379 | 0.0571 | 0.2160 |
| Intercept-only reference | 0.4638 | 0.0631 | 0.0619 | 0.2438 |

The original elastic-net search used `C ∈ {0.01, 0.1, 1}` and `L1-ratio ∈ {0, 0.25, 0.5, 0.75, 1}`. **20 of 25 outer folds selected `C = 0.1` and `L1-ratio = 0`**, i.e. the ridge/L2 end of the path. This supported retaining the full predictor set with shrinkage rather than imposing a hard sparse subset.

## End-to-end workflow

The repository now exposes the full technical workflow rather than only the final model:

1. **Raw-data audit and deterministic cleaning** — schema validation, encoded missing values, invalid Killip codes, unit-entry corrections, duplicate checks.
2. **Exploratory analysis** — outcome prevalence, missingness and data-quality summaries.
3. **Leakage-safe preprocessing** — imputation, scaling and categorical encoding fitted inside training folds only.
4. **Model development** — penalised logistic regression, standard logistic regression, Random Forest, Gradient Boosting and an intercept-only reference.
5. **Hyperparameter tuning** — four-fold stratified inner cross-validation using log loss.
6. **Internal validation** — five-fold stratified outer cross-validation repeated five times, yielding 25 genuine held-out folds.
7. **Class-imbalance ablation** — no rebalancing vs class weighting vs random oversampling, with resampling confined to training folds.
8. **Performance evaluation** — ROC-AUC, PR-AUC, Brier score, log loss, calibration intercept/slope and repeated-split stability.
9. **Uncertainty** — 2,000 patient-level bootstrap resamples of the final out-of-fold predictions.
10. **Sensitivity analyses** — ordinal vs categorical Killip coding and missingness indicators.
11. **Threshold analysis** — sensitivity, specificity, PPV, NPV and proportion flagged at 5%, 10%, 15% and 20% risk thresholds.
12. **Decision-curve analysis** — net benefit against treat-all and treat-none reference strategies.
13. **Interpretation** — shrunk coefficients/odds ratios plus raw-predictor permutation importance evaluated in held-out outer folds.
14. **Reproducibility** — fixed random seed, saved metadata, model pipeline and exported analysis tables/figures.

## Data-quality decisions

Cleaning was deliberately limited to rules supported by the supplied data dictionary or unmistakable unit-entry errors:

| Raw issue | Treatment |
|---|---|
| `Hypothension` header | renamed to `Hypotension` |
| `Hyperthension` header | renamed to `Hypertension` |
| `Hypothension = Unknown` | recoded as missing |
| `Killip_class = -1` | recoded as missing because valid classes are 1–4 |
| `Height = 1.75` | corrected to 175 cm |
| `Height = 1690` | corrected to 169 cm |

After recoding there were **24 missing predictor cells** and **no exact duplicate rows**. No patient was removed.

## Leakage-safe preprocessing

- Continuous/ordinal predictors: median imputation + `StandardScaler`.
- Binary predictors: most-frequent-value imputation.
- Smoking: most-frequent-value imputation + one-hot encoding with the first level dropped.
- All transformations are fitted **inside each training fold**.
- Random oversampling, when tested, is also restricted to training folds.

This prevents validation-fold information from leaking into preprocessing, resampling or tuning.

## Class-imbalance ablation

The 6.62% mortality rate makes class imbalance important, but imbalance correction was treated as an empirical question rather than an automatic preprocessing step.

| Strategy | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| **No rebalancing** | **0.7881** | **0.2783** | **0.0556** | **0.2070** |
| Class weighting | 0.7746 | 0.2712 | 0.1711 | 0.5138 |
| Random oversampling | 0.7703 | 0.2700 | 0.1700 | 0.5108 |

Both rebalancing strategies worsened probability accuracy substantially, so the natural class distribution was retained.

## Sensitivity analyses

| Specification | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| Primary ridge | 0.7881 | 0.2783 | 0.0556 | 0.2070 |
| Killip as categorical | 0.7786 | 0.2233 | 0.0567 | 0.2103 |
| Missing indicators added | 0.7889 | 0.2787 | 0.0556 | 0.2069 |

The ordinal treatment of Killip class was retained. Missingness indicators added essentially no practical improvement, so the simpler imputation-only specification was preferred.

## Threshold trade-offs

These thresholds are **illustrative, not recommended treatment thresholds**.

| Risk threshold | Sensitivity | Specificity | PPV | NPV | Flagged high risk |
|---|---:|---:|---:|---:|---:|
| 5% | 78.8% | 61.9% | 12.8% | 97.6% | 40.8% |
| 10% | 53.8% | 83.1% | 18.4% | 96.2% | 19.4% |
| 15% | 38.5% | 92.5% | 26.7% | 95.5% | 9.6% |
| 20% | 23.1% | 95.6% | 27.3% | 94.6% | 5.6% |

Decision-curve net benefit for ridge at the same thresholds was **0.0335**, **0.0181**, **0.0131** and **0.0051**, respectively.

## Model interpretation

The final full-data ridge model used `C = 0.1`. Because continuous/ordinal predictors are standardised, their odds ratios correspond approximately to a **one-standard-deviation increase**, not a raw one-unit increase.

Largest coefficients by absolute magnitude included:

| Predictor | Coefficient | Odds ratio |
|---|---:|---:|
| Age | 0.6554 | 1.9258 |
| Heart-rate indicator | 0.4691 | 1.5985 |
| Time to relief >1 hour | 0.3208 | 1.3783 |
| Previous myocardial infarction | 0.3139 | 1.3687 |
| Killip class | 0.3072 | 1.3596 |
| Anterior infarct location | 0.2750 | 1.3165 |

Held-out raw-predictor permutation analysis identified **Age** as the dominant predictive contributor, followed by **Killip class**, **heart-rate indicator**, **time to relief**, and **weight**. These are predictive associations, not causal effects.

## Repository structure

```text
.
├── ami_patient_data.csv                         # supplied course dataset
├── AMI_Mortality_Prediction.ipynb              # legacy Jan-2026 notebook (kept for history)
├── Technical_Validation_Report.pdf             # legacy Jan-2026 report (kept for history)
├── notebooks/
│   └── ISDS_Option_A_Full_Workflow.ipynb       # current step-by-step notebook
├── src/
│   └── isds_option_a_pipeline.py               # current reproducible full pipeline
├── docs/
│   ├── METHODOLOGY.md                          # modelling and validation design
│   ├── RESULTS.md                              # current validated results
│   └── LEGACY_ARTIFACTS.md                     # explains old vs current analysis
└── requirements.txt
```

The old root notebook/report are retained only as historical artifacts. **Do not use their AUC/Brier values as the current project result.** The current methodological specification and headline numbers are those documented above and in `docs/RESULTS.md`.

## How to reproduce the current workflow

```bash
git clone https://github.com/Tanjim-hossain/Prediction-Model-for-30-Day-Mortality-Following-Acute-Myocardial-Infarction.git
cd Prediction-Model-for-30-Day-Mortality-Following-Acute-Myocardial-Infarction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/isds_option_a_pipeline.py --data ami_patient_data.csv --output outputs/current
```

The full pipeline is computationally heavier than the old notebook because it performs repeated nested cross-validation, model tuning, ablations, bootstrap uncertainty and held-out permutation importance.

## Reproducibility settings

- Random seed: `2026`
- Outer validation: stratified 5-fold CV × 5 repeats
- Inner validation: stratified 4-fold CV
- Tuning criterion: log loss
- Final model: ridge logistic regression
- Final `C`: `0.1`
- Class rebalancing: none
- Bootstrap resamples: 2,000
- scikit-learn version used for the submitted analysis: `1.6.1`

## Skills demonstrated

`Python` · `pandas` · `scikit-learn` · `imbalanced-learn` · EDA · data cleaning · missing-data handling · feature encoding · regularisation · logistic regression · Random Forest · Gradient Boosting · nested cross-validation · hyperparameter tuning · class-imbalance ablation · calibration · bootstrap uncertainty · decision-curve analysis · permutation importance · TRIPOD-oriented reporting · reproducible ML workflows

## Limitations

The project has only **52 events**, so uncertainty remains substantial. Validation is internal only; transportability to another hospital, country, period or treatment setting is unknown. The source assignment did not provide full recruitment/measurement metadata. Thresholds are illustrative. Coefficients and permutation importance are predictive rather than causal. External validation would be required before clinical use.

---

**Author:** Tanjim Hossain  
**Programme:** Master of Statistics and Data Science — Data Science, Hasselt University
