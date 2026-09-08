#!/usr/bin/env python3
"""Validate the complete executed analysis notebook and optionally extract figures.

The notebook is treated as the authoritative computational record. This utility
checks that the expected executed structure is present and verifies that key
analysis stages have not been removed during repository maintenance.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

EXPECTED_COUNTS = {
    "cells": 72,
    "code_cells": 32,
    "executed_code_cells": 32,
    "outputs": 126,
}

KEY_ANALYSIS_SIGNATURES = {
    "elastic_net": ("elasticnet", "l1_ratio"),
    "ridge_logistic": ("ridge", "logisticregression"),
    "standard_logistic": ("standard logistic", "logisticregression"),
    "random_forest": ("randomforestclassifier",),
    "gradient_boosting": ("gradientboostingclassifier",),
    "class_weighting": ("class_weight",),
    "random_oversampling": ("randomoversampler",),
    "nested_validation": ("outer", "inner", "stratified"),
    "bootstrap_uncertainty": ("bootstrap", "2000"),
    "decision_curve": ("decision curve", "net benefit"),
    "permutation_importance": ("permutation", "importance"),
    "model_serialization": ("joblib.dump",),
}

# Cell indices correspond to the supplied fully executed notebook.
FIGURE_CELLS = {
    19: "outcome_and_missingness.png",
    49: "calibration_curve.png",
    66: "decision_curve.png",
    68: "permutation_importance.png",
}


def _all_text(nb: dict) -> str:
    chunks: list[str] = []
    for cell in nb.get("cells", []):
        chunks.append("".join(cell.get("source", [])))
        for output in cell.get("outputs", []):
            text = output.get("text")
            if isinstance(text, list):
                chunks.append("".join(text))
            elif text is not None:
                chunks.append(str(text))
            for key, value in output.get("data", {}).items():
                if key.startswith("text/"):
                    chunks.append("".join(value) if isinstance(value, list) else str(value))
    return "\n".join(chunks).lower()


def _count_structure(nb: dict) -> dict[str, int]:
    cells = nb.get("cells", [])
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    executed = [cell for cell in code_cells if cell.get("execution_count") is not None]
    return {
        "cells": len(cells),
        "code_cells": len(code_cells),
        "executed_code_cells": len(executed),
        "outputs": sum(len(cell.get("outputs", [])) for cell in cells),
    }


def _extract_figures(nb: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for cell_index, filename in FIGURE_CELLS.items():
        try:
            cell = nb["cells"][cell_index]
        except IndexError as exc:
            raise RuntimeError(f"Expected figure cell {cell_index} is missing") from exc

        payload = None
        for output in cell.get("outputs", []):
            image = output.get("data", {}).get("image/png")
            if image is not None:
                payload = "".join(image) if isinstance(image, list) else image
                break

        if payload is None:
            raise RuntimeError(f"No embedded PNG found in expected cell {cell_index}")

        (output_dir / filename).write_bytes(base64.b64decode(payload))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--extract-figures", type=Path, default=None)
    args = parser.parse_args()

    nb = json.loads(args.notebook.read_text(encoding="utf-8"))

    observed = _count_structure(nb)
    if observed != EXPECTED_COUNTS:
        raise SystemExit(
            "Notebook structure mismatch. "
            f"Expected {EXPECTED_COUNTS}, observed {observed}."
        )

    text = _all_text(nb)
    missing = [
        name
        for name, tokens in KEY_ANALYSIS_SIGNATURES.items()
        if not all(token in text for token in tokens)
    ]
    if missing:
        raise SystemExit(f"Missing expected analysis stages: {', '.join(missing)}")

    if args.extract_figures is not None:
        _extract_figures(nb, args.extract_figures)

    print("Complete notebook audit passed")
    print(f"Structure: {observed}")
    print(f"Verified stages: {', '.join(KEY_ANALYSIS_SIGNATURES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
