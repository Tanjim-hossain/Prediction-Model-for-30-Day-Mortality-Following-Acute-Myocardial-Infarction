# Model Artifacts

The complete executed notebook fits the selected full-data ridge pipeline after model selection and writes two deployment-readiness artifacts:

```text
final_ridge_prediction_pipeline.joblib
final_ridge_model_metadata.json
```

These generated files are intentionally not version-controlled in this repository. The serialized `joblib` object is environment-dependent and uses Python pickle semantics, so it should only be loaded from a trusted source and ideally be regenerated in the target environment.

To reproduce the packaged model, run the complete reusable pipeline:

```bash
python src/mortality_prediction_pipeline.py \
  --data ami_patient_data.csv \
  --output outputs/current
```

The generated model and metadata are written beneath the chosen output directory. See [`docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md) for the packaging boundary and inference considerations.
