from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mortality_prediction_pipeline as pipeline


def _minimal_frame():
    return pd.DataFrame(
        {
            "Day30_mortality": [0, 1, 0],
            "Gender": [0, 1, np.nan],
            "Age": [50.0, 70.0, 60.0],
            "Killip_class": [1, -1, 2],
            "Diabetes": [0, 1, 0],
            "Hypothension": [0, "Unknown", 1],
            "Heart_rate": [0, 1, 0],
            "Anterior_infarct_location": [0, 1, 0],
            "Previous_myocardial_infarction": [0, 1, 0],
            "Height": [1.75, 1690.0, 170.0],
            "Weight": [70.0, 80.0, 75.0],
            "Hyperthension": [0, 1, 0],
            "Smoking": [1, 2, 3],
            "Hypercholesterolaemia": [0, 1, 0],
            "Previous_angina_pectoris": [0, 1, 0],
            "Family_history_of_MI": [0, 1, 0],
            "ST_elevation_leads": [2, 4, 3],
            "Time_To_Relief": [0, 1, 0],
        }
    )


def test_cleaning_rules(tmp_path):
    path = tmp_path / "mini.csv"
    _minimal_frame().to_csv(path, index=False)
    raw, clean = pipeline.load_and_clean(path)

    assert "Hypothension" in raw.columns
    assert "Hypotension" in clean.columns
    assert "Hypertension" in clean.columns
    assert pd.isna(clean.loc[1, "Hypotension"])
    assert pd.isna(clean.loc[1, "Killip_class"])
    assert clean.loc[0, "Height"] == 175.0
    assert clean.loc[1, "Height"] == 169.0


def test_threshold_metrics_are_bounded():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    result = pipeline.threshold_metrics(y, p, 0.5)

    for key in ["sensitivity", "specificity", "PPV", "NPV", "flagged_high_risk"]:
        assert 0.0 <= result[key] <= 1.0


def test_primary_preprocessor_fits_with_missing_values():
    clean = _minimal_frame().rename(
        columns={"Hypothension": "Hypotension", "Hyperthension": "Hypertension"}
    )
    clean["Hypotension"] = pd.to_numeric(
        clean["Hypotension"].replace("Unknown", np.nan), errors="coerce"
    )
    clean.loc[clean["Killip_class"] == -1, "Killip_class"] = np.nan
    clean.loc[clean["Height"] == 1.75, "Height"] = 175.0
    clean.loc[clean["Height"] == 1690, "Height"] = 169.0

    X = clean.drop(columns=[pipeline.OUTCOME])
    transformed = pipeline.make_preprocessor().fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert np.isfinite(np.asarray(transformed, dtype=float)).all()
