# Reproducibility

## Complete computational record

The primary source of truth is:

`notebooks/01_Complete_Executed_Analysis.ipynb`

The notebook contains 72 cells, 32 code cells, 32/32 executed code cells and 126 preserved outputs. It includes the full data audit, cleaning, preprocessing, model development, repeated nested validation, class-imbalance experiments, candidate-model comparison, calibration, bootstrap uncertainty, sensitivity analyses, threshold analysis, decision-curve analysis, permutation importance and model packaging.

## Environment

The executed analysis records:

- Python 3.12.13
- NumPy 2.0.2
- pandas 2.3.3
- matplotlib 3.10.0
- seaborn 0.13.2
- scikit-learn 1.6.1

The main runtime dependencies are pinned or constrained in `requirements.txt`.

## Randomness

Primary random seed: `2026`.

Deterministic fold/permutation seeds are also used where repeated random operations are required.

## Validation settings

- outer CV: 5 stratified folds × 5 repeats;
- inner CV: 4 stratified folds;
- tuning objective: log loss;
- bootstrap resamples: 2,000;
- permutation repeats: 20 per predictor per outer split.

## Run the reusable pipeline

```bash
python src/mortality_prediction_pipeline.py \
  --data ami_patient_data.csv \
  --output outputs/current
```

## Run the Elastic Net development screen

```bash
python src/elastic_net_screen.py \
  --data ami_patient_data.csv \
  --output outputs/elastic_net
```

## Run from the companion notebook

Open:

`notebooks/02_Reproducible_Workflow.ipynb`

This notebook calls the same reusable source module while keeping the original fully executed notebook unchanged as the analytical record.

## Notebook integrity audit

```bash
python tools/audit_complete_notebook.py \
  notebooks/01_Complete_Executed_Analysis.ipynb
```

The audit verifies the expected notebook structure and checks that the major analytical stages are still present.

## Tests

```bash
pytest -q
```

CI installs dependencies, checks Python syntax, executes the unit tests and validates the complete executed notebook on pull requests and pushes to `main`.

## Result traceability

The repository stores:

- the complete executed notebook;
- the reusable prediction pipeline;
- the Elastic Net development screen;
- machine-readable summary tables;
- original figures extracted from the executed notebook;
- methodology, results, model-card and deployment-readiness documentation;
- tests and CI configuration.

The complete notebook preserves its original Kaggle runtime paths and execution outputs because those are part of the historical computational record. For a fresh repository-based run, use the reusable source module or `02_Reproducible_Workflow.ipynb`.
