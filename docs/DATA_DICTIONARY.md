# Data dictionary

The supplied Option A dataset contains one binary outcome and 17 candidate predictors.

| Variable | Meaning / coding |
|---|---|
| `Day30_mortality` | 30-day mortality (0/1) |
| `Gender` | Female gender (0/1) |
| `Age` | Age in years; assignment range 19–110 |
| `Killip_class` | Killip class, a measure of left-ventricular function; classes 1–4 |
| `Diabetes` | Diabetes (0/1) |
| `Hypotension` | Hypotension, systolic blood pressure <100 (0/1) |
| `Heart_rate` | Tachycardia indicator, pulse >80 (0/1) |
| `Anterior_infarct_location` | Anterior infarct location (0/1) |
| `Previous_myocardial_infarction` | Previous myocardial infarction (0/1) |
| `Height` | Height in centimetres; assignment range 140–212 |
| `Weight` | Weight in kilograms; assignment range 36–213 |
| `Hypertension` | Hypertension history (0/1) |
| `Smoking` | 1=never, 2=ex-smoker, 3=current smoker |
| `Hypercholesterolaemia` | Hypercholesterolaemia (0/1) |
| `Previous_angina_pectoris` | Previous angina pectoris (0/1) |
| `Family_history_of_MI` | Family history of myocardial infarction (0/1) |
| `ST_elevation_leads` | Number of ECG leads with ST elevation; assignment range 0–11 |
| `Time_To_Relief` | Time to relief of chest pain >1 hour (0/1) |

## Data-quality rules used in the analysis

The raw file is conceptually preserved. The processed analysis copy applies only the following deterministic corrections:

- `Hypothension` → `Hypotension`
- `Hyperthension` → `Hypertension`
- `Hypotension = "Unknown"` → missing
- `Killip_class = -1` → missing
- `Height = 1.75` → 175 cm
- `Height = 1690` → 169 cm

No patient is deleted. Imputation is estimated inside model-training folds.
