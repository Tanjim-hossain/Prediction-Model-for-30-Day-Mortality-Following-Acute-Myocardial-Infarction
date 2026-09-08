"""Reproducible ISDS Option A analysis pipeline.

Prediction of 30-day mortality after acute myocardial infarction.

This script mirrors the current project report: auditable cleaning, leakage-safe
preprocessing, repeated nested internal validation, model comparison, class-imbalance
ablations, calibration, bootstrap uncertainty, sensitivity analyses, threshold summaries,
decision-curve analysis, final-model coefficients, and held-out permutation importance.

Run from the repository root:
    python src/isds_option_a_pipeline.py --data ami_patient_data.csv --output outputs/current

The project is an academic/internal-validation exercise and is not a clinical tool.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 2026
OUTCOME = "Day30_mortality"
OUTER_SPLITS = 5
OUTER_REPEATS = 5
INNER_SPLITS = 4
BOOTSTRAPS = 2000
PERMUTATION_REPEATS = 20

CONTINUOUS_ORDINAL = ["Age", "Killip_class", "Height", "Weight", "ST_elevation_leads"]
BINARY = [
    "Gender",
    "Diabetes",
    "Hypotension",
    "Heart_rate",
    "Anterior_infarct_location",
    "Previous_myocardial_infarction",
    "Hypertension",
    "Hypercholesterolaemia",
    "Previous_angina_pectoris",
    "Family_history_of_MI",
    "Time_To_Relief",
]
CATEGORICAL = ["Smoking"]


def load_and_clean(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw data and apply only pre-specified, auditable corrections."""
    raw = pd.read_csv(path)
    data = raw.copy()

    data = data.rename(columns={"Hypothension": "Hypotension", "Hyperthension": "Hypertension"})
    data["Hypotension"] = data["Hypotension"].replace("Unknown", np.nan)
    data["Hypotension"] = pd.to_numeric(data["Hypotension"], errors="coerce")
    data.loc[data["Killip_class"] == -1, "Killip_class"] = np.nan
    data.loc[data["Height"] == 1.75, "Height"] = 175.0
    data.loc[data["Height"] == 1690, "Height"] = 169.0

    expected = {OUTCOME, *CONTINUOUS_ORDINAL, *BINARY, *CATEGORICAL}
    missing_columns = sorted(expected.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")
    if not set(data[OUTCOME].dropna().unique()).issubset({0, 1}):
        raise ValueError(f"{OUTCOME} must be binary 0/1")

    return raw, data


def audit_table(raw: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("patients", len(data)),
        ("candidate_predictors", data.shape[1] - 1),
        ("deaths", int(data[OUTCOME].sum())),
        ("event_rate", float(data[OUTCOME].mean())),
        ("exact_duplicate_rows", int(data.duplicated().sum())),
        ("missing_predictor_cells_after_recoding", int(data.drop(columns=[OUTCOME]).isna().sum().sum())),
        ("raw_columns", raw.shape[1]),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def make_preprocessor() -> ColumnTransformer:
    continuous_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    binary_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("continuous_ordinal", continuous_pipe, CONTINUOUS_ORDINAL),
            ("binary", binary_pipe, BINARY),
            ("categorical", categorical_pipe, CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_outer_cv() -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=RANDOM_SEED,
    )


def nested_oof(
    model,
    grid: Dict[str, Iterable],
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate genuine held-out predictions under repeated nested CV."""
    predictions = []
    tuning = []

    for split_id, (train_idx, test_idx) in enumerate(make_outer_cv().split(X, y), 1):
        repeat = (split_id - 1) // OUTER_SPLITS + 1
        fold = (split_id - 1) % OUTER_SPLITS + 1
        inner_cv = StratifiedKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED + split_id,
        )
        search = GridSearchCV(
            clone(model),
            grid,
            scoring="neg_log_loss",
            cv=inner_cv,
            refit=True,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = search.predict_proba(X.iloc[test_idx])[:, 1]

        for local_i, original_i in enumerate(test_idx):
            predictions.append(
                {
                    "patient_index": X.index[original_i],
                    "repeat": repeat,
                    "fold": fold,
                    "y_true": int(y.iloc[original_i]),
                    "predicted_probability": float(prob[local_i]),
                }
            )
        tuning.append(
            {
                "repeat": repeat,
                "fold": fold,
                "best_parameters": json.dumps(search.best_params_, sort_keys=True),
                "inner_log_loss": float(-search.best_score_),
            }
        )

    repeated = pd.DataFrame(predictions)
    patient = repeated.groupby("patient_index", as_index=False).agg(
        y_true=("y_true", "first"),
        predicted_probability=("predicted_probability", "mean"),
    )
    return repeated, patient, pd.DataFrame(tuning)


def metrics(patient_oof: pd.DataFrame) -> Dict[str, float]:
    yy = patient_oof["y_true"].to_numpy()
    pp = patient_oof["predicted_probability"].to_numpy()
    return {
        "ROC_AUC": float(roc_auc_score(yy, pp)),
        "PR_AUC": float(average_precision_score(yy, pp)),
        "Brier": float(brier_score_loss(yy, pp)),
        "Log_loss": float(log_loss(yy, pp)),
    }


def repeat_metrics(repeated_oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for repeat, grp in repeated_oof.groupby("repeat"):
        m = metrics(grp)
        m["repeat"] = int(repeat)
        rows.append(m)
    return pd.DataFrame(rows).sort_values("repeat")


def build_models(preprocessor: ColumnTransformer):
    ridge = Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    class_weight=None,
                    max_iter=20000,
                    tol=1e-4,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    weighted = Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=20000,
                    tol=1e-4,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    oversampled = ImbPipeline(
        [
            ("preprocessor", clone(preprocessor)),
            ("oversampler", RandomOverSampler(random_state=RANDOM_SEED)),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    class_weight=None,
                    max_iter=20000,
                    tol=1e-4,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    standard_logistic = Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                LogisticRegression(
                    penalty=None,
                    solver="lbfgs",
                    max_iter=20000,
                    tol=1e-4,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    rf = Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_SEED,
                    class_weight=None,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    gb = Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            ("model", GradientBoostingClassifier(random_state=RANDOM_SEED)),
        ]
    )

    return {
        "Ridge logistic": (ridge, {"model__C": [0.01, 0.1, 1.0]}),
        "Standard logistic": (standard_logistic, {}),
        "Random Forest": (
            rf,
            {
                "model__n_estimators": [300, 600],
                "model__max_depth": [3, 5, None],
                "model__min_samples_leaf": [2, 5, 10],
                "model__max_features": ["sqrt", 0.7],
            },
        ),
        "Gradient Boosting": (
            gb,
            {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.03, 0.05, 0.10],
                "model__max_depth": [1, 2, 3],
                "model__min_samples_leaf": [5, 10],
            },
        ),
        "Ridge class-weighted": (weighted, {"model__C": [0.01, 0.1, 1.0]}),
        "Ridge oversampled": (oversampled, {"model__C": [0.01, 0.1, 1.0]}),
    }


def null_oof(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for split_id, (train_idx, test_idx) in enumerate(make_outer_cv().split(X, y), 1):
        model = DummyClassifier(strategy="prior")
        model.fit(np.zeros((len(train_idx), 1)), y.iloc[train_idx])
        prob = model.predict_proba(np.zeros((len(test_idx), 1)))[:, 1]
        repeat = (split_id - 1) // OUTER_SPLITS + 1
        fold = (split_id - 1) % OUTER_SPLITS + 1
        for j, original_i in enumerate(test_idx):
            rows.append(
                {
                    "patient_index": X.index[original_i],
                    "repeat": repeat,
                    "fold": fold,
                    "y_true": int(y.iloc[original_i]),
                    "predicted_probability": float(prob[j]),
                }
            )
    repeated = pd.DataFrame(rows)
    patient = repeated.groupby("patient_index", as_index=False).agg(
        y_true=("y_true", "first"),
        predicted_probability=("predicted_probability", "mean"),
    )
    return repeated, patient


def calibration_summary(patient_oof: pd.DataFrame) -> Dict[str, float]:
    y = patient_oof["y_true"].to_numpy()
    p = patient_oof["predicted_probability"].to_numpy()
    p_safe = np.clip(p, 1e-6, 1 - 1e-6)
    logit_p = np.log(p_safe / (1 - p_safe)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=20000)
    model.fit(logit_p, y)
    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0, 0]),
        "mean_predicted_risk": float(p.mean()),
        "observed_event_rate": float(y.mean()),
    }


def bootstrap_intervals(patient_oof: pd.DataFrame, n_boot: int = BOOTSTRAPS) -> pd.DataFrame:
    y = patient_oof["y_true"].to_numpy()
    p = patient_oof["predicted_probability"].to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        yy, pp = y[idx], p[idx]
        if np.unique(yy).size < 2:
            continue
        values.append(
            [
                roc_auc_score(yy, pp),
                average_precision_score(yy, pp),
                brier_score_loss(yy, pp),
                log_loss(yy, pp),
            ]
        )
    arr = np.asarray(values)
    q = np.quantile(arr, [0.025, 0.975], axis=0)
    point = metrics(patient_oof)
    names = ["ROC_AUC", "PR_AUC", "Brier", "Log_loss"]
    return pd.DataFrame(
        {
            "metric": names,
            "estimate": [point[n] for n in names],
            "lower_95": q[0],
            "upper_95": q[1],
        }
    )


def make_killip_categorical_preprocessor() -> ColumnTransformer:
    continuous_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    binary_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    killip_cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    categories=[[1, 2, 3], [1, 2, 3, 4]],
                    drop="first",
                    handle_unknown="error",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("continuous", continuous_pipe, ["Age", "Height", "Weight", "ST_elevation_leads"]),
            ("binary", binary_pipe, BINARY),
            ("categorical", killip_cat_pipe, ["Smoking", "Killip_class"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_missing_indicator_preprocessor() -> ColumnTransformer:
    primary = make_preprocessor()
    transformers = list(primary.transformers)
    transformers.append(
        (
            "missing_indicators",
            MissingIndicator(features="all", error_on_new=False),
            CONTINUOUS_ORDINAL + BINARY + CATEGORICAL,
        )
    )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def threshold_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "flagged_high_risk": float(pred.mean()),
    }


def decision_curve(y_true: np.ndarray, prob: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    n = len(y_true)
    prevalence = float(np.mean(y_true))
    rows = []
    for pt in thresholds:
        pred = (prob >= pt).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        odds = pt / (1 - pt)
        nb_model = (tp / n) - (fp / n) * odds
        nb_all = prevalence - (1 - prevalence) * odds
        rows.append(
            {
                "threshold": float(pt),
                "ridge_net_benefit": float(nb_model),
                "treat_all_net_benefit": float(nb_all),
                "treat_none_net_benefit": 0.0,
            }
        )
    return pd.DataFrame(rows)


def plot_eda(data: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = data[OUTCOME].value_counts().sort_index()
    ax.bar(["Survived (0)", "Died (1)"], counts.values)
    ax.set_ylabel("Patients")
    ax.set_title("30-day mortality class distribution")
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value}\n({value/len(data):.1%})", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=180)
    plt.close(fig)

    missing = data.drop(columns=[OUTCOME]).isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(missing.index[::-1], missing.values[::-1])
    ax.set_xlabel("Missing cells")
    ax.set_title("Missing values after auditable recoding")
    fig.tight_layout()
    fig.savefig(out / "missingness_after_recoding.png", dpi=180)
    plt.close(fig)


def plot_calibration(oof_map: Dict[str, pd.DataFrame], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    for name, df in oof_map.items():
        y = df["y_true"].to_numpy()
        p = df["predicted_probability"].to_numpy()
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=7, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0, 0.4)
    ax.set_xlabel("Mean predicted 30-day mortality risk")
    ax.set_ylabel("Observed 30-day mortality proportion")
    ax.set_title("Internal-validation calibration curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "calibration_curves.png", dpi=180)
    plt.close(fig)


def plot_dca(dca: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(dca["threshold"] * 100, dca["ridge_net_benefit"], label="Ridge prediction model")
    ax.plot(dca["threshold"] * 100, dca["treat_all_net_benefit"], linestyle="--", label="Treat all")
    ax.plot(dca["threshold"] * 100, dca["treat_none_net_benefit"], linestyle=":", label="Treat none")
    ax.set_xlabel("Risk threshold (%)")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve — 30-day mortality prediction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "decision_curve.png", dpi=180)
    plt.close(fig)


def final_coefficients(X: pd.DataFrame, y: pd.Series, preprocessor: ColumnTransformer) -> Tuple[Pipeline, pd.DataFrame]:
    model = Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    C=0.1,
                    class_weight=None,
                    max_iter=20000,
                    tol=1e-4,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    model.fit(X, y)
    names = model.named_steps["preprocessor"].get_feature_names_out()
    coef = model.named_steps["model"].coef_[0]
    table = pd.DataFrame({"predictor": names, "coefficient": coef, "odds_ratio": np.exp(coef)})
    table["abs_coefficient"] = table["coefficient"].abs()
    table = table.sort_values("abs_coefficient", ascending=False).drop(columns="abs_coefficient")
    return model, table


def heldout_raw_permutation_importance(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
) -> pd.DataFrame:
    """Shuffle each raw predictor in each outer holdout fold and score log-loss increase."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for split_id, (train_idx, test_idx) in enumerate(make_outer_cv().split(X, y), 1):
        model = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        solver="lbfgs",
                        C=0.1,
                        class_weight=None,
                        max_iter=20000,
                        tol=1e-4,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        X_test = X.iloc[test_idx].copy()
        y_test = y.iloc[test_idx].to_numpy()
        base_p = model.predict_proba(X_test)[:, 1]
        base_loss = log_loss(y_test, base_p)
        repeat = (split_id - 1) // OUTER_SPLITS + 1
        fold = (split_id - 1) % OUTER_SPLITS + 1

        for col in X.columns:
            increases = []
            for _ in range(PERMUTATION_REPEATS):
                shuffled = X_test.copy()
                shuffled[col] = rng.permutation(shuffled[col].to_numpy())
                p = model.predict_proba(shuffled)[:, 1]
                increases.append(log_loss(y_test, p) - base_loss)
            rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "predictor": col,
                    "mean_log_loss_increase": float(np.mean(increases)),
                }
            )
    result = pd.DataFrame(rows)
    return (
        result.groupby("predictor", as_index=False)["mean_log_loss_increase"]
        .mean()
        .sort_values("mean_log_loss_increase", ascending=False)
    )


def plot_permutation_importance(importance: pd.DataFrame, out: Path) -> None:
    top = importance.head(12).sort_values("mean_log_loss_increase")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["predictor"], top["mean_log_loss_increase"])
    ax.set_xlabel("Mean increase in held-out log loss after permutation")
    ax.set_title("Final ridge — outer-fold raw-predictor permutation importance")
    fig.tight_layout()
    fig.savefig(out / "permutation_importance.png", dpi=180)
    plt.close(fig)


def run(data_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    models_dir = output_dir / "models"
    for d in (tables_dir, figures_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    raw, data = load_and_clean(data_path)
    audit_table(raw, data).to_csv(tables_dir / "data_audit.csv", index=False)
    data.isna().sum().rename("missing_cells").to_csv(tables_dir / "missingness.csv")
    plot_eda(data, figures_dir)

    X = data.drop(columns=[OUTCOME])
    y = data[OUTCOME].astype(int)
    preprocessor = make_preprocessor()

    model_map = build_models(preprocessor)
    oof_by_name: Dict[str, pd.DataFrame] = {}
    repeated_by_name: Dict[str, pd.DataFrame] = {}
    performance_rows = []

    for name, (model, grid) in model_map.items():
        repeated, patient, tuning = nested_oof(model, grid, X, y)
        slug = name.lower().replace(" ", "_").replace("-", "_")
        repeated.to_csv(tables_dir / f"{slug}_repeated_oof.csv", index=False)
        patient.to_csv(tables_dir / f"{slug}_patient_oof.csv", index=False)
        tuning.to_csv(tables_dir / f"{slug}_tuning.csv", index=False)
        performance_rows.append({"model": name, **metrics(patient)})
        oof_by_name[name] = patient
        repeated_by_name[name] = repeated

    null_rep, null_patient = null_oof(X, y)
    null_rep.to_csv(tables_dir / "intercept_only_repeated_oof.csv", index=False)
    null_patient.to_csv(tables_dir / "intercept_only_patient_oof.csv", index=False)
    performance_rows.append({"model": "Intercept-only reference", **metrics(null_patient)})
    pd.DataFrame(performance_rows).to_csv(tables_dir / "candidate_model_performance.csv", index=False)

    ridge_rep = repeated_by_name["Ridge logistic"]
    ridge_oof = oof_by_name["Ridge logistic"]
    repeat_metrics(ridge_rep).to_csv(tables_dir / "ridge_repeat_metrics.csv", index=False)

    pd.DataFrame([calibration_summary(ridge_oof)]).to_csv(
        tables_dir / "ridge_calibration_summary.csv", index=False
    )
    bootstrap_intervals(ridge_oof).to_csv(tables_dir / "ridge_bootstrap_95ci.csv", index=False)

    ridge_grid = {"model__C": [0.01, 0.1, 1.0]}
    killip_model = Pipeline(
        [
            ("preprocessor", make_killip_categorical_preprocessor()),
            (
                "model",
                LogisticRegression(
                    penalty="l2", solver="lbfgs", max_iter=20000, tol=1e-4, random_state=RANDOM_SEED
                ),
            ),
        ]
    )
    _, killip_oof, _ = nested_oof(killip_model, ridge_grid, X, y)
    mi_model = Pipeline(
        [
            ("preprocessor", make_missing_indicator_preprocessor()),
            (
                "model",
                LogisticRegression(
                    penalty="l2", solver="lbfgs", max_iter=20000, tol=1e-4, random_state=RANDOM_SEED
                ),
            ),
        ]
    )
    _, mi_oof, _ = nested_oof(mi_model, ridge_grid, X, y)
    pd.DataFrame(
        [
            {"specification": "Primary ridge", **metrics(ridge_oof)},
            {"specification": "Killip categorical", **metrics(killip_oof)},
            {"specification": "Missing indicators added", **metrics(mi_oof)},
        ]
    ).to_csv(tables_dir / "sensitivity_analyses.csv", index=False)

    yy = ridge_oof["y_true"].to_numpy()
    pp = ridge_oof["predicted_probability"].to_numpy()
    pd.DataFrame([threshold_metrics(yy, pp, t) for t in [0.05, 0.10, 0.15, 0.20]]).to_csv(
        tables_dir / "threshold_metrics.csv", index=False
    )
    dca = decision_curve(yy, pp, np.linspace(0.01, 0.30, 60))
    dca.to_csv(tables_dir / "decision_curve.csv", index=False)
    plot_dca(dca, figures_dir)

    plot_calibration(
        {
            "Ridge logistic": oof_by_name["Ridge logistic"],
            "Standard logistic": oof_by_name["Standard logistic"],
            "Random Forest": oof_by_name["Random Forest"],
            "Gradient Boosting": oof_by_name["Gradient Boosting"],
        },
        figures_dir,
    )

    final_model, coefficients = final_coefficients(X, y, preprocessor)
    coefficients.to_csv(tables_dir / "final_ridge_coefficients.csv", index=False)
    joblib.dump(final_model, models_dir / "final_ridge_pipeline.joblib")

    importance = heldout_raw_permutation_importance(X, y, preprocessor)
    importance.to_csv(tables_dir / "outer_fold_permutation_importance.csv", index=False)
    plot_permutation_importance(importance, figures_dir)

    summary = {
        "random_seed": RANDOM_SEED,
        "patients": int(len(data)),
        "deaths": int(y.sum()),
        "event_rate": float(y.mean()),
        "outer_validation": f"{OUTER_SPLITS}-fold stratified CV x {OUTER_REPEATS} repeats",
        "inner_validation": f"{INNER_SPLITS}-fold stratified CV",
        "tuning_metric": "log loss",
        "selected_model": "ridge logistic regression",
        "selected_C": 0.1,
        "class_rebalancing": "none",
        "note": "Internal validation only; not for clinical deployment.",
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Finished. Outputs written to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full ISDS Option A prediction workflow.")
    parser.add_argument("--data", type=Path, default=Path("ami_patient_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/current"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data, args.output)
