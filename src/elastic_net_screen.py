"""Reproduce the initial elastic-net development screen from the final executed notebook.

This step precedes the matched ridge experiments. It tunes both C and l1_ratio inside
4-fold inner cross-validation and evaluates the selected model in repeated 5-fold outer
cross-validation. The purpose is to show why the final penalised model is ridge/L2.

Usage:
    python src/elastic_net_screen.py --data ami_patient_data.csv --output outputs/elastic_net
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from mortality_prediction_pipeline import RANDOM_SEED, OUTCOME, load_and_clean, make_preprocessor

OUTER_SPLITS = 5
OUTER_REPEATS = 5
INNER_SPLITS = 4


def run(data_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _, data = load_and_clean(data_path)
    X = data.drop(columns=[OUTCOME])
    y = data[OUTCOME].astype(int)

    pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor()),
            (
                "model",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    class_weight=None,
                    max_iter=30000,
                    tol=1e-4,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    grid = {
        "model__C": [0.01, 0.1, 1.0],
        "model__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
    }
    outer = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=RANDOM_SEED,
    )

    rows = []
    tuning = []
    for split_id, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        repeat = (split_id - 1) // OUTER_SPLITS + 1
        fold = (split_id - 1) % OUTER_SPLITS + 1
        inner = StratifiedKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED + split_id,
        )
        search = GridSearchCV(
            clone(pipeline),
            grid,
            scoring="neg_log_loss",
            cv=inner,
            refit=True,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = search.predict_proba(X.iloc[test_idx])[:, 1]

        for local_i, original_i in enumerate(test_idx):
            rows.append(
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
                "best_C": float(search.best_params_["model__C"]),
                "best_l1_ratio": float(search.best_params_["model__l1_ratio"]),
                "inner_cv_log_loss": float(-search.best_score_),
                "training_deaths": int(y.iloc[train_idx].sum()),
                "test_deaths": int(y.iloc[test_idx].sum()),
            }
        )
        print(
            f"outer fold {split_id:02d}/25 | repeat={repeat}, fold={fold} | "
            f"C={search.best_params_['model__C']} | "
            f"l1_ratio={search.best_params_['model__l1_ratio']}"
        )

    repeated = pd.DataFrame(rows)
    patient = repeated.groupby("patient_index", as_index=False).agg(
        y_true=("y_true", "first"),
        predicted_probability=("predicted_probability", "mean"),
    )
    yy = patient["y_true"].to_numpy()
    pp = patient["predicted_probability"].to_numpy()
    metrics = {
        "ROC_AUC": float(roc_auc_score(yy, pp)),
        "PR_AUC": float(average_precision_score(yy, pp)),
        "Brier_score": float(brier_score_loss(yy, pp)),
        "Log_loss": float(log_loss(yy, pp)),
    }

    tuning_df = pd.DataFrame(tuning)
    frequency = (
        tuning_df.groupby(["best_C", "best_l1_ratio"], as_index=False)
        .size()
        .rename(columns={"size": "outer_folds_selected"})
        .sort_values("outer_folds_selected", ascending=False)
    )

    repeated.to_csv(output_dir / "elastic_net_repeated_oof.csv", index=False)
    patient.to_csv(output_dir / "elastic_net_patient_oof.csv", index=False)
    tuning_df.to_csv(output_dir / "elastic_net_tuning.csv", index=False)
    frequency.to_csv(output_dir / "elastic_net_selection_frequency.csv", index=False)
    (output_dir / "elastic_net_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("\nInternally validated performance")
    print(pd.Series(metrics).round(4))
    print("\nSelected hyperparameter frequency")
    print(frequency.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the elastic-net development screen.")
    parser.add_argument("--data", type=Path, default=Path("ami_patient_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/elastic_net"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data, args.output)
