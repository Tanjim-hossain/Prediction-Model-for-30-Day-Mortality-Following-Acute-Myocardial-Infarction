# 30-Day Mortality Prediction after Acute Myocardial Infarction

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Validation](https://img.shields.io/badge/Validation-Repeated%20Nested%20CV-2F855A)](#validation-design)
[![Status](https://img.shields.io/badge/Status-Internal%20Validation%20Only-6B7280)](#limitations-and-responsible-use)
[![CI](https://github.com/Tanjim-hossain/ami-30day-mortality-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Tanjim-hossain/ami-30day-mortality-prediction/actions/workflows/ci.yml)

**Clinical prediction · statistical learning · leakage-safe validation · reproducible machine learning**

This repository presents an end-to-end prediction-modelling study for estimating **30-day mortality risk after acute myocardial infarction (AMI)**. The project was completed for the *Inference for Statistics and Data Science* course at Hasselt University and has been organised here as a professional, reproducible data-science portfolio project.

> **Plain-language summary:** using routinely available variables in the supplied AMI dataset, the project estimates which patients are at higher risk of death within 30 days. The selected model is a regularised logistic regression. It shows useful internal discrimination and good probability calibration, but it has **not** been externally validated and is **not a clinical decision tool**.

## Project at a glance

| Item | Value |
|---|---|
| Prediction target | 30-day mortality after AMI |
| Patients | **785** |
| Deaths | **52 (6.62%)** |
| Candidate predictors | **17** |
| Final model | **Ridge-penalised logistic regression** |
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

## What this project demonstrates

The repository exposes the complete modelling lifecycle rather than only the final estimator:

- raw-data schema and quality auditing;
- deterministic, traceable data cleaning;
- leakage-safe imputation, scaling and categorical encoding;
- elastic-net regularisation for feature/complexity control;
- ridge and standard logistic regression;
- Random Forest and Gradient Boosting benchmarks;
- repeated nested cross-validation and hyperparameter tuning;
- class-weighting and random-oversampling ablations;
- ROC-AUC, PR-AUC, Brier score and log-loss evaluation;
- calibration intercept/slope and calibration diagnostics;
- 2,000-sample bootstrap uncertainty;
- sensitivity analyses for predictor representation and missingness;
- threshold operating-characteristic trade-offs;
- decision-curve analysis;
- penalised coefficients and held-out permutation importance;
- reusable Python source code, automated tests, CI and machine-readable results.

### Start here

- **Portfolio analysis notebook:** [`notebooks/01_Full_Analysis_Executed.ipynb`](notebooks/01_Full_Analysis_Executed.ipynb) — a GitHub-optimised presentation built directly from the supplied fully executed Kaggle notebook, retaining the full analytical narrative and key validated outputs.
- **Clean reproducible notebook:** [`notebooks/02_Reproducible_Workflow.ipynb`](notebooks/02_Reproducible_Workflow.ipynb) — script-backed workflow for re-running the current pipeline.
- **Reusable pipeline:** [`src/isds_option_a_pipeline.py`](src/isds_option_a_pipeline.py).
- **Elastic-net development screen:** [`src/elastic_net_screen.py`](src/elastic_net_screen.py).
- **Technical report:** [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md).
- **Machine-readable results:** [`results/tables/`](results/tables/).

---

## Modelling workflow

```mermaid
flowchart LR
    A[Raw AMI data] --> B[Schema & quality audit]
    B --> C[Auditable cleaning]
    C --> D[Leakage-safe preprocessing]
    D --> E[Repeated nested CV]
    E --> F[Elastic-net development]
    F --> G[Candidate model comparison]
    G --> H[Ridge selected]
    H --> I[Calibration + bootstrap uncertainty]
    I --> J[Sensitivity + imbalance ablations]
    J --> K[Thresholds + decision curve]
    K --> L[Coefficients + held-out permutation importance]
```

## Data audit and cleaning

The supplied dataset contains 785 patients, 52 deaths and 17 candidate predictors. The raw audit found:

- **7** explicitly missing source cells;
- **3** `Hypothension` values encoded as `Unknown`;
- **14** `Killip_class` values encoded as `-1`;
- **2** implausible height entries (`1.75` and `1690`);
- **0** exact duplicate rows;
- **0** duplicate predictor profiles.

Cleaning was deliberately conservative. Invalid codes were recoded as missing, the height values were corrected to 175 cm and 169 cm, two misspelled headers were normalised, and **no patient was removed**. After recoding there were 24 missing predictor cells.

![Outcome imbalance and missingness](docs/assets/data_audit.svg)

Because only 52 events are available, overfitting and unstable single-split estimates are material risks. Raw accuracy is also uninformative for a 6.62% event rate. The project therefore emphasises regularisation, repeated internal validation and proper probability-scoring metrics.

## Leakage-safe preprocessing

Every transformation that can learn from data is fitted **inside the relevant training fold**.

| Predictor group | Primary handling |
|---|---|
| Continuous / ordinal | Median imputation + standardisation |
| Binary | Most-frequent-value imputation |
| Smoking | Most-frequent imputation + one-hot encoding |
| Killip class | Ordinal in the primary analysis |
| Oversampling | Training folds only, only in the ablation experiment |

This prevents validation information from leaking into imputation statistics, scaling parameters, category encoding, resampling or hyperparameter selection.

## Validation design

The primary evaluation uses repeated nested cross-validation:

- **outer loop:** 5 stratified folds × 5 repeats = 25 held-out folds;
- **inner loop:** 4 stratified folds;
- **tuning criterion:** log loss;
- **random seed:** 2026.

Each patient receives one genuine out-of-fold probability in every outer repeat. The five OOF probabilities are averaged for the headline patient-level metrics, while repeat-level metrics are retained to assess stability.

## Elastic-net development and why ridge was selected

The initial penalised logistic search used:

- `C ∈ {0.01, 0.1, 1.0}`;
- `l1_ratio ∈ {0, 0.25, 0.5, 0.75, 1}`.

**20 of 25 outer folds selected `C=0.1` and `l1_ratio=0`**, i.e. the ridge/L2 endpoint. The initial elastic-net model achieved ROC-AUC **0.7878** and log loss **0.2071**. The analysis therefore retained all predictors with shrinkage rather than imposing a hard sparse subset and used a stable ridge specification for the matched final experiments.

## Candidate-model comparison

![Candidate model comparison](docs/assets/model_comparison.svg)

| Model | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| **Ridge logistic** | **0.7881** | 0.2783 | **0.0556** | **0.2070** |
| Random Forest | 0.7806 | 0.2462 | 0.0564 | 0.2094 |
| Standard logistic | 0.7726 | **0.2816** | 0.0558 | 0.2127 |
| Gradient Boosting | 0.7509 | 0.2379 | 0.0571 | 0.2160 |
| Intercept-only reference | 0.4638 | 0.0631 | 0.0619 | 0.2438 |

Ridge was retained for the best **overall** balance of discrimination, proper scoring-rule performance, calibration and repeated-validation stability. The decision was not based on ROC-AUC alone.

## Class-imbalance ablation

Mortality prevalence is low, but imbalance correction was tested rather than assumed to be beneficial.

| Ridge strategy | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| **No rebalancing** | **0.7881** | **0.2783** | **0.0556** | **0.2070** |
| Class weighting | 0.7746 | 0.2712 | 0.1711 | 0.5138 |
| Random oversampling | 0.7703 | 0.2700 | 0.1700 | 0.5108 |

Both rebalancing approaches substantially worsened probability accuracy, so the final model preserves the natural class distribution.

## Calibration and uncertainty

The final ridge model achieved:

- calibration intercept **0.1277**;
- calibration slope **1.0580**;
- mean predicted mortality risk **6.63%**;
- observed mortality **6.62%**.

![Calibration summary](docs/assets/calibration.svg)

Uncertainty for the final OOF prediction set was estimated using **2,000 patient-level bootstrap resamples**.

| Metric | Estimate | 95% bootstrap CI |
|---|---:|---:|
| ROC-AUC | **0.7881** | 0.7236–0.8478 |
| PR-AUC | **0.2783** | 0.1741–0.4054 |
| Brier score | **0.0556** | 0.0423–0.0690 |
| Log loss | **0.2070** | 0.1658–0.2487 |

## Sensitivity analyses

| Specification | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| Primary ridge | 0.7881 | 0.2783 | 0.0556 | 0.2070 |
| Killip as categorical | 0.7786 | 0.2233 | 0.0567 | 0.2103 |
| Missing indicators added | 0.7889 | 0.2787 | 0.0556 | 0.2069 |

The primary ordinal Killip specification was retained. Missingness indicators produced essentially no practical improvement, favouring the simpler fold-specific imputation strategy.

## Threshold trade-offs

No single threshold was optimised on the development data. Instead, several illustrative risk thresholds make the operating trade-off transparent.

| Risk threshold | Sensitivity | Specificity | PPV | NPV | Flagged high risk |
|---|---:|---:|---:|---:|---:|
| 5% | 78.8% | 61.9% | 12.8% | 97.6% | 40.8% |
| 10% | 53.8% | 83.1% | 18.4% | 96.2% | 19.4% |
| 15% | 38.5% | 92.5% | 26.7% | 95.5% | 9.6% |
| 20% | 23.1% | 95.6% | 27.3% | 94.6% | 5.6% |

These thresholds are descriptive and are **not treatment recommendations**.

## Decision-curve analysis

![Decision curve](docs/assets/decision_curve.svg)

Ridge net benefit at 5%, 10%, 15% and 20% thresholds was **0.0335, 0.0181, 0.0131 and 0.0051**. In the original explored grid the model had greater estimated net benefit than both reference strategies over approximately 1%–30%. This does not establish causal treatment benefit from using the model.

## Model interpretation

The final full-data ridge model is refitted only after model selection. Performance claims remain based on held-out OOF predictions.

| Predictor | Coefficient | Odds ratio |
|---|---:|---:|
| Age | 0.6554 | 1.9258 |
| Heart-rate indicator | 0.4691 | 1.5985 |
| Time to relief >1 hour | 0.3208 | 1.3783 |
| Previous myocardial infarction | 0.3139 | 1.3687 |
| Killip class | 0.3072 | 1.3596 |
| Anterior infarct location | 0.2750 | 1.3165 |

Continuous/ordinal predictors were standardised, so their odds ratios correspond approximately to a **one-standard-deviation increase**, not a raw one-unit increase.

Held-out raw-predictor permutation importance ranked **Age** first, followed by **Killip class**, **heart-rate indicator**, **time to relief**, and **weight**.

![Outer-fold permutation importance](docs/assets/permutation_importance.svg)

Importance and coefficients are predictive summaries, not causal effects.

## Repository structure

```text
.
├── README.md
├── ami_patient_data.csv
├── requirements.txt
├── Makefile
├── pyproject.toml
├── CITATION.cff
├── .gitignore
├── .github/workflows/ci.yml
├── notebooks/
│   ├── 01_Full_Analysis_Executed.ipynb
│   └── 02_Reproducible_Workflow.ipynb
├── src/
│   ├── isds_option_a_pipeline.py
│   └── elastic_net_screen.py
├── results/
│   ├── README.md
│   └── tables/                         # curated validated outputs
├── docs/
│   ├── TECHNICAL_REPORT.md
│   ├── METHODOLOGY.md
│   ├── RESULTS.md
│   ├── DATA_DICTIONARY.md
│   ├── DATA_USAGE.md
│   ├── MODEL_CARD.md
│   ├── TRIPOD_MAPPING.md
│   ├── LEGACY_ARTIFACTS.md
│   └── assets/                         # current portfolio visuals
└── tests/test_pipeline.py
```

## Reproduce the analysis

```bash
git clone https://github.com/Tanjim-hossain/ami-30day-mortality-prediction.git
cd ami-30day-mortality-prediction

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Full current pipeline
python src/isds_option_a_pipeline.py --data ami_patient_data.csv --output outputs/current

# Initial elastic-net development screen
python src/elastic_net_screen.py --data ami_patient_data.csv --output outputs/elastic_net

# Quality checks
pytest -q
```

Convenience commands are also available through `make install`, `make test`, `make run` and `make elastic-net`.

## Technical stack

**Python 3.12.13** · **NumPy 2.0.2** · **pandas 2.3.3** · **scikit-learn 1.6.1** · **matplotlib 3.10.0** · **seaborn 0.13.2** · **imbalanced-learn** · **joblib** · **Jupyter/Kaggle**

## Skills demonstrated

`EDA` · `data-quality auditing` · `missing-data handling` · `feature encoding` · `regularisation` · `logistic regression` · `Random Forest` · `Gradient Boosting` · `nested cross-validation` · `hyperparameter tuning` · `class-imbalance ablation` · `calibration` · `bootstrap uncertainty` · `decision-curve analysis` · `permutation importance` · `TRIPOD-oriented reporting` · `reproducible ML pipelines` · `testing/CI`

## Documentation

- [Technical report](docs/TECHNICAL_REPORT.md)
- [Methodology](docs/METHODOLOGY.md)
- [Validated results](docs/RESULTS.md)
- [Data dictionary](docs/DATA_DICTIONARY.md)
- [Dataset provenance / reuse note](docs/DATA_USAGE.md)
- [Model card](docs/MODEL_CARD.md)
- [Assignment / TRIPOD traceability](docs/TRIPOD_MAPPING.md)
- [Legacy-analysis note](docs/LEGACY_ARTIFACTS.md)
- [Curated result tables](results/tables/)

## Limitations and responsible use

This model was developed from a single supplied dataset with only 52 outcome events. All performance estimates are **internal-validation estimates**; there is no external, temporal or independent-site validation. Transportability to other hospitals, countries, treatment eras or patient populations is unknown.

The analysis is predictive rather than causal. Coefficients, odds ratios and permutation importance should not be interpreted as treatment effects or biological mechanisms. Threshold and decision-curve analyses are exploratory and do not define a clinical policy.

**Do not use this repository for direct patient-care decisions.**

---

**Author:** Tanjim Hossain  
**Programme:** MSc Statistics and Data Science — Data Science, Hasselt University  
**Course project:** Inference for Statistics and Data Science, Option A — Prediction Modelling
