# Freight Rate Prediction

Predict `posted_rate` ($) for freight loads from route, equipment, weight, date, and market signals.

## Quick start

```bash
python -m pip install -r requirements.txt
python run_pipeline.py
```

This trains the model, writes:

- `validation_predictions.csv` (12,000 rows: `load_id,predicted_rate`)
- `december_chart_predictions.csv` (31 fixed-lane December rates)
- `scorer_results/candidate_december.png` (via `score.py`)

## Data

| File | Role |
|------|------|
| `data/train-test.csv` | 48k labeled loads (Jan–Oct 2025) |
| `data/validation.csv` | 12k unlabeled loads (Nov–Dec 2025) |
| `data/validation-predictions-template.csv` | `load_id` template `predict.py` fills in |
| `data/december-chart-inputs.csv` | Fixed Lexington→Fort Wayne lane for Dec chart |

**Getting the data:** `data/*.csv` and the assessment PDF are gitignored on purpose — they're Spotter's materials, not mine to redistribute in a public repo. Drop your copies of the four provided CSVs into `data/` (same filenames as the assessment ZIP) before running `run_pipeline.py`.

## Approach (short)

- **Split:** chronological — train Jan–Aug, hold out Sep–Oct (mirrors Nov–Dec test)
- **Cleaning:** abs(negative weights); impute weight/market_index; handle unseen cities
- **Features:** temporal cycles, geo (haversine/bearing), market×distance interactions, smoothed lane RPM encodings
- **Model:** predict **$/mile**, then × distance. Blend:
  - 50% LightGBM (MAE)
  - 30% CatBoost (MAE)
  - 20% Extra Trees
- **Holdout MAE:** ~$111.5 (~4.9% MAPE)

Full write-up: `REPORT.md` (source) / `Freight_Rate_Report.pdf` (with charts)

## Layout

```
src/features.py   cleaning + feature engineering
src/train.py      time-split validation + final training
src/predict.py    validation + December inference
run_pipeline.py   end-to-end entrypoint
score.py          Spotter-provided validator / chart
```
