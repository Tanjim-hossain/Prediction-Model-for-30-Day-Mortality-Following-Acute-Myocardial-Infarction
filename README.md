# 30-Day Mortality Prediction after Acute Myocardial Infarction

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Validation](https://img.shields.io/badge/Validation-Repeated%20Nested%20CV-2F855A)](#validation-design)
[![Status](https://img.shields.io/badge/Status-Internal%20Validation%20Only-6B7280)](#limitations-and-responsible-use)
[![CI](https://github.com/Tanjim-hossain/ami-30day-mortality-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Tanjim-hossain/ami-30day-mortality-prediction/actions/workflows/ci.yml)

**Clinical prediction · statistical learning · leakage-safe validation · reproducible machine learning**

This repository contains the complete development and internal validation of a model for estimating **30-day mortality risk after acute myocardial infarction (AMI)**. The analysis moves from raw-data auditing and deterministic cleaning through regularised statistical learning, tree-based machine-learning benchmarks, repeated nested cross-validation, probability calibration, uncertainty analysis, sensitivity checks, decision-curve analysis, model interpretation and final model packaging.

> **Plain-language summary:** the model uses 17 patient-level predictors to estimate the probability of death within 30 days after AMI. Ridge-penalised logistic regression provided the strongest overall internally validated performance, with ROC-AUC **0.7881**, Brier score **0.0556** and calibration slope **1.0580**. The model has not been externally validated and must not be used as a direct clinical decision tool.

## Project at a glance

| Item | Result |
|---|---|
| Prediction target | 30-day mortality after AMI |
| Patients | **785** |
| Deaths | **52 (6.62%)** |
| Candidate predictors | **17** |
| Selected model | **Ridge-penalised logistic regression** |
| Regularisation | **C = 0.1** |
| Class rebalancing | **None** |
| Internal validation | **5-fold stratified outer CV × 5 repeats** |
| Hyperparameter tuning | **4-fold stratified inner CV** |
| Tuning objective | **Log loss** |
| ROC-AUC | **0.7881** |
| PR-AUC | **0.2783** |
| Brier score | **0.0556** |
| Log loss | **0.2070** |
| Calibration intercept / slope | **0.1277 / 1.0580** |

## Complete computational record

The primary analytical artifact is the fully executed notebook:

**[`notebooks/01_Complete_Executed_Analysis.ipynb`](notebooks/01_Complete_Executed_Analysis.ipynb)**

It is intentionally preserved as the complete computational record rather than reduced to a showcase notebook:

- **72 total cells**
- **32 code cells**
- **32/32 code cells executed**
- **126 preserved outputs**
- complete data audit and cleaning
- Elastic Net development screen
- ridge, standard logistic, Random Forest and Gradient Boosting models
- class-weighting and random-oversampling experiments
- repeated nested cross-validation and tuning outputs
- calibration and 2,000-resample bootstrap uncertainty
- sensitivity analyses
- intercept-only reference model
- threshold operating characteristics
- original decision-curve analysis
- outer-fold permutation importance
- final pipeline serialization and model metadata

The notebook retains its original Kaggle runtime paths and execution outputs because those are part of the executed record. For a fresh repository-based run, use [`notebooks/02_Reproducible_Workflow.ipynb`](notebooks/02_Reproducible_Workflow.ipynb) or the reusable source module [`src/mortality_prediction_pipeline.py`](src/mortality_prediction_pipeline.py).

## End-to-end workflow

```mermaid
flowchart LR
    A[Raw AMI data] --> B[Schema & quality audit]
    B --> C[Deterministic cleaning]
    C --> D[Leakage-safe preprocessing]
    D --> E[Repeated nested CV]
    E --> F[Elastic Net development]
    F --> G[Candidate model comparison]
    G --> H[Ridge selection]
    H --> I[Calibration + bootstrap uncertainty]
    I --> J[Sensitivity + imbalance ablations]
    J --> K[Thresholds + decision curve]
    K --> L[Interpretation + model packaging]
```

## 1. Data audit and exploratory analysis

The raw dataset contains 785 patients and 17 candidate predictors. Only 52 patients died within 30 days, producing an event prevalence of **6.62%**. This class imbalance makes raw accuracy a poor evaluation target and increases the risk of unstable single-split estimates.

The data-quality audit identified:

- **7** explicitly missing source cells;
- **3** hypotension values encoded as `Unknown`;
- **14** `Killip_class = -1` values;
- **2** implausible height entries (`1.75` and `1690`);
- **0** exact duplicate rows;
- **0** duplicate predictor profiles.

![Outcome distribution and missingness](results/figures/outcome_and_missingness.png)

### Deterministic cleaning

| Raw issue | Processing decision |
|---|---|
| `Hypothension` | rename to `Hypotension` |
| `Hyperthension` | rename to `Hypertension` |
| `Hypotension = Unknown` | recode as missing |
| `Killip_class = -1` | recode as missing |
| `Height = 1.75` | correct to 175 cm |
| `Height = 1690` | correct to 169 cm |

No patient was deleted. After recoding, **24 predictor cells** were missing.

## 2. Leakage-safe preprocessing

All transformations that learn from data are fitted **inside the relevant training fold**.

| Predictor group | Processing |
|---|---|
| Continuous / ordinal | median imputation + standardisation |
| Binary | most-frequent-value imputation |
| Smoking | most-frequent imputation + one-hot encoding |
| Killip class | ordinal in the primary analysis |
| Random oversampling | training folds only, in the ablation experiment |

This prevents information from validation folds from leaking into imputation statistics, scaling parameters, category encoding, resampling or hyperparameter selection.

## 3. Validation design

The primary performance estimates come from **repeated nested stratified cross-validation**:

- outer loop: 5 folds × 5 repeats = **25 held-out folds**;
- inner loop: **4 stratified folds**;
- hyperparameter selection criterion: **log loss**;
- primary random seed: **2026**.

Each patient receives one genuine out-of-fold probability per outer repeat. The five probabilities are averaged for the headline patient-level estimates, while repeat-level metrics are retained to examine stability.

## 4. Elastic Net development and ridge selection

The initial penalised logistic model tuned both regularisation strength and penalty mixture:

- `C ∈ {0.01, 0.1, 1.0}`
- `l1_ratio ∈ {0, 0.25, 0.5, 0.75, 1}`

**20 of 25 outer folds selected `C = 0.1` and `l1_ratio = 0`**, corresponding to the ridge/L2 endpoint. The Elastic Net development model achieved approximately:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.7878 |
| PR-AUC | 0.2790 |
| Brier score | 0.0556 |
| Log loss | 0.2071 |

The result supported keeping the complete predictor set with shrinkage rather than forcing a sparse L1-selected subset.

## 5. Candidate-model comparison

The common validation framework evaluated ridge logistic regression, standard logistic regression, Random Forest, Gradient Boosting and an intercept-only prevalence reference.

![Candidate-model comparison](docs/assets/model_comparison.svg)

| Model | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| **Ridge logistic** | **0.7881** | 0.2783 | **0.0556** | **0.2070** |
| Random Forest | 0.7806 | 0.2462 | 0.0564 | 0.2094 |
| Standard logistic | 0.7726 | **0.2816** | 0.0558 | 0.2127 |
| Gradient Boosting | 0.7509 | 0.2379 | 0.0571 | 0.2160 |
| Intercept-only reference | 0.4638 | 0.0631 | 0.0619 | 0.2438 |

Ridge was retained for its overall balance of discrimination, proper probability-scoring performance, calibration and repeated-validation stability. The selection was not based on ROC-AUC alone.

### Repeat-wise ridge stability

- ROC-AUC: **0.7831 ± 0.0056**
- PR-AUC: **0.2610 ± 0.0310**
- Brier score: **0.0559 ± 0.0009**
- log loss: **0.2084 ± 0.0026**

## 6. Class-imbalance ablation

Low event prevalence does not automatically imply that rebalancing improves probability prediction, so three matched ridge strategies were compared.

| Strategy | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| **No rebalancing** | **0.7881** | **0.2783** | **0.0556** | **0.2070** |
| Class weighting | 0.7746 | 0.2712 | 0.1711 | 0.5138 |
| Random oversampling | 0.7703 | 0.2700 | 0.1700 | 0.5108 |

Both rebalancing approaches substantially degraded Brier score and log loss. The final model therefore preserves the natural class distribution.

## 7. Calibration and uncertainty

The selected ridge model achieved:

- calibration intercept: **0.1277**
- calibration slope: **1.0580**
- mean predicted risk: **6.63%**
- observed mortality: **6.62%**

![Original calibration curve from the executed notebook](results/figures/calibration_curve.png)

Metric uncertainty was assessed using **2,000 patient-level bootstrap resamples** of the final out-of-fold prediction set.

| Metric | Estimate | 95% bootstrap interval |
|---|---:|---:|
| ROC-AUC | **0.7881** | 0.7236–0.8478 |
| PR-AUC | **0.2783** | 0.1741–0.4054 |
| Brier score | **0.0556** | 0.0423–0.0690 |
| Log loss | **0.2070** | 0.1658–0.2487 |

## 8. Sensitivity analyses

Two modelling assumptions were challenged explicitly.

| Specification | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| Primary ridge | 0.7881 | 0.2783 | 0.0556 | 0.2070 |
| Killip class categorical | 0.7786 | 0.2233 | 0.0567 | 0.2103 |
| Missing indicators added | 0.7889 | 0.2787 | 0.0556 | 0.2069 |

The ordinal Killip representation was retained. Explicit missingness indicators provided essentially no practical improvement over the simpler fold-specific imputation strategy.

## 9. Threshold operating characteristics

No single operating threshold was selected from the development data. Instead, several illustrative risk thresholds show the sensitivity-specificity trade-off.

| Risk threshold | Sensitivity | Specificity | PPV | NPV | Flagged high risk |
|---|---:|---:|---:|---:|---:|
| 5% | 78.8% | 61.9% | 12.8% | 97.6% | 40.8% |
| 10% | 53.8% | 83.1% | 18.4% | 96.2% | 19.4% |
| 15% | 38.5% | 92.5% | 26.7% | 95.5% | 9.6% |
| 20% | 23.1% | 95.6% | 27.3% | 94.6% | 5.6% |

These values are descriptive operating characteristics, not treatment recommendations.

## 10. Decision-curve analysis

The figure below is the **original decision-curve output preserved in the executed notebook**, not a redrawn substitute.

![Original decision curve from the executed notebook](results/figures/decision_curve.png)

Ridge net benefit at 5%, 10%, 15% and 20% thresholds was **0.0335, 0.0181, 0.0131 and 0.0051**, respectively. Across the evaluated 1%–30% threshold grid, the model showed positive estimated net benefit relative to the reference strategies over the relevant range. This remains an internally evaluated prediction analysis and does not establish causal benefit from acting on the model.

## 11. Model interpretation

The final ridge pipeline is refitted on all 785 patients only after the modelling strategy has been selected. Performance claims remain based on held-out out-of-fold predictions.

Largest absolute ridge coefficients include:

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

Continuous and ordinal variables were standardised, so their odds ratios correspond approximately to a **one-standard-deviation increase**, not a one-unit raw-scale change.

Outer-fold raw-predictor permutation importance identified **Age** as the dominant contributor, followed by **Killip class**, **heart-rate indicator**, **time to relief** and **weight**.

![Original permutation-importance plot from the executed notebook](results/figures/permutation_importance.png)

Coefficients, odds ratios and permutation importance are predictive summaries rather than causal effects.

## 12. Model packaging and deployment readiness

After model selection, the full preprocessing-plus-ridge pipeline is fitted on all available data and serialized with `joblib`. The executed notebook writes:

```text
final_ridge_prediction_pipeline.joblib
final_ridge_model_metadata.json
```

This demonstrates **model packaging and inference readiness**. It does not claim a production API, web application, real-time clinical integration, monitoring system or regulatory validation. Generated model artifacts are intentionally reproducible rather than committed as trusted binaries; see [`models/README.md`](models/README.md) and [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

## Repository structure

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── pyproject.toml
├── Makefile
├── ami_patient_data.csv
├── notebooks/
│   ├── 01_Complete_Executed_Analysis.ipynb   # complete executed record
│   └── 02_Reproducible_Workflow.ipynb       # clean script-backed rerun
├── src/
│   ├── mortality_prediction_pipeline.py      # reusable end-to-end pipeline
│   └── elastic_net_screen.py                 # Elastic Net development screen
├── results/
│   ├── figures/                              # original executed figures
│   └── tables/                               # machine-readable summaries
├── models/
│   └── README.md                             # generated model-artifact guidance
├── docs/
│   ├── TECHNICAL_REPORT.md
│   ├── METHODOLOGY.md
│   ├── RESULTS.md
│   ├── DATA_DICTIONARY.md
│   ├── DATA_USAGE.md
│   ├── MODEL_CARD.md
│   ├── DEPLOYMENT.md
│   ├── REPRODUCIBILITY.md
│   └── TRIPOD_MAPPING.md
├── tools/
│   └── audit_complete_notebook.py
└── tests/
    └── test_pipeline.py
```

## Reproduce the analysis

```bash
git clone https://github.com/Tanjim-hossain/ami-30day-mortality-prediction.git
cd ami-30day-mortality-prediction

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Reusable full pipeline
python src/mortality_prediction_pipeline.py \
  --data ami_patient_data.csv \
  --output outputs/current

# Initial Elastic Net development screen
python src/elastic_net_screen.py \
  --data ami_patient_data.csv \
  --output outputs/elastic_net

# Unit tests
pytest -q

# Verify that the complete executed notebook has not been reduced
python tools/audit_complete_notebook.py \
  notebooks/01_Complete_Executed_Analysis.ipynb
```

Convenience commands are also available through `make install`, `make test`, `make audit-notebook`, `make run` and `make elastic-net`.

## Technical stack

**Python 3.12.13** · **NumPy 2.0.2** · **pandas 2.3.3** · **scikit-learn 1.6.1** · **matplotlib 3.10.0** · **seaborn 0.13.2** · **imbalanced-learn** · **joblib** · **Jupyter/Kaggle**

## Documentation

- [Complete executed analysis](notebooks/01_Complete_Executed_Analysis.ipynb)
- [Technical report](docs/TECHNICAL_REPORT.md)
- [Methodology](docs/METHODOLOGY.md)
- [Validated results](docs/RESULTS.md)
- [Data dictionary](docs/DATA_DICTIONARY.md)
- [Dataset use and provenance](docs/DATA_USAGE.md)
- [Model card](docs/MODEL_CARD.md)
- [Model packaging and deployment readiness](docs/DEPLOYMENT.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Prediction-model reporting traceability](docs/TRIPOD_MAPPING.md)
- [Machine-readable results](results/)

## What this work demonstrates

`data-quality auditing` · `EDA` · `missing-data handling` · `feature encoding` · `regularisation` · `Elastic Net` · `ridge logistic regression` · `Random Forest` · `Gradient Boosting` · `nested cross-validation` · `hyperparameter tuning` · `class-imbalance ablation` · `calibration` · `bootstrap uncertainty` · `sensitivity analysis` · `decision-curve analysis` · `permutation importance` · `model serialization` · `reproducible ML pipelines` · `testing` · `CI`

## Limitations and responsible use

This model was developed from a single dataset with only 52 outcome events. All performance estimates are **internal-validation estimates**; there is no external, temporal or independent-site validation. Transportability to other hospitals, countries, treatment eras or patient populations is unknown.

The analysis is predictive rather than causal. Coefficients, odds ratios and permutation importance should not be interpreted as treatment effects or biological mechanisms. Threshold and decision-curve analyses are exploratory and do not define a clinical policy.

**Do not use this repository for direct patient-care decisions.**

The software is released under the MIT License. Dataset reuse is addressed separately in [`docs/DATA_USAGE.md`](docs/DATA_USAGE.md); the software licence does not grant independent rights to the underlying data.

---

**Author:** Tanjim Hossain
