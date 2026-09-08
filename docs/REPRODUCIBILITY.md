# Reproducibility

## Complete computational record

The primary source of truth is:

`notebooks/01_Complete_Executed_Analysis.ipynb`

It contains 72 cells, including 32 executed code cells and all preserved outputs from the complete analysis.

## Environment

The executed analysis records:

- Python 3.12
- NumPy 2.0.2
- pandas 2.3.3
- matplotlib 3.10.0
- seaborn 0.13.2
- scikit-learn 1.6.1

The repository pins or constrains the main runtime dependencies in `requirements.txt`.

## Randomness

Primary random seed: `2026`.

The analysis also uses deterministic fold/permutation seeds where repeated random operations are required.

## Validation settings

- outer CV: 5 stratified folds × 5 repeats;
- inner CV: 4 stratified folds;
- tuning objective: log loss;
- bootstrap resamples: 2,000;
- permutation repeats: 20 per predictor per outer split.

## Run the reusable pipeline

```bash
python src/isds_option_a_pipeline.py \
  --data ami_patient_data.csv \
  --output outputs/current
```

## Run the Elastic Net screen

```bash
python src/elastic_net_screen.py \
  --data ami_patient_data.csv \
  --output outputs/elastic_net
```

## Tests

```bash
pytest -q
```

CI installs dependencies, checks Python syntax and executes the test suite on pull requests and pushes to `main`.

## Result traceability

The repository stores:

- the complete executed notebook;
- machine-readable summary tables;
- original notebook figures;
- methodology and result documentation;
- model packaging code;
- tests and CI configuration.
