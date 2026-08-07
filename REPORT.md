# Freight Rate ML Assessment — Report

## 1. Problem

Predict the posted freight rate ($) for each load using route geography, equipment type, shipment weight, calendar date, and market signals (`market_index`, `quote_signal`).

Development data covers **Jan–Oct 2025** (48k loads). The submission set covers **Nov–Dec 2025** (12k loads). A secondary task prices a fixed Lexington → Fort Wayne Dry Van lane for every day in December 2025.

## 2. Data exploration — key findings

- **Distance dominates price.** Correlation of `distance` with `posted_rate` ≈ **0.91**.
- **Rate-per-mile is more stable** than raw dollars (median ≈ $2.15/mi). Equipment premiums: Reefer > Flatbed > Dry Van.
- **Market index is mostly a daily factor** (within-day std ≈ 0.025), with clear seasonality through the year.
- **Validation includes cities unseen in training**, so encodings must generalize.
- Train ends in October; validation is November–December → this is a **future prediction** problem.

## 3. Data-quality issues and fixes

| Issue | Treatment |
|-------|-----------|
| ~292 negative `weight` values | Treated as sign-flip errors → `abs(weight)`; flag retained |
| Missing `weight` | Median by equipment type |
| Missing `market_index` | Daily median → month median → global median |
| December chart lacks lat/lon, market_index, quote_signal | City coords from training map; Dec `market_index` from validation daily medians; lane quote prior |

## 4. Feature engineering

- **Temporal:** month, day-of-week, day-of-year, weekend flags, sin/cos cyclical encodings
- **Geographic:** Δlat/Δlon, haversine miles, bearing, distance÷haversine, log/sqrt distance
- **Interactions:** `distance×market`, `distance×quote`, `expected_base_rate = distance×quote×market`
- **Target encodings (train-only, smoothed):** lane / pickup / delivery historical rate-per-mile
- **Categoricals:** `equipment`, `pickup`, `delivery`

## 5. Train / validation split

Chronological split (not random):

- **Train:** 2025-01-01 → 2025-08-31
- **Holdout:** 2025-09-01 → 2025-10-31

After holdout evaluation, models were **retrained on all labeled data** for submission.

## 6. Model choice

Experiments compared Random Forest, Extra Trees, LightGBM, XGBoost, CatBoost, linear models, and blends.

**Winning approach:** predict **rate-per-mile**, then multiply by distance. Optimize MAE. Final blend:

| Component | Weight |
|-----------|--------|
| LightGBM (MAE on $/mi) | 50% |
| CatBoost (MAE on $/mi) | 30% |
| Extra Trees (on $/mi) | 20% |

### Holdout results (Sep–Oct)

| Model | MAE ($) | MAPE % | R² |
|-------|---------|--------|-----|
| Final blend | **~111.5** | **~4.9** | ~0.827 |
| LightGBM RPM only | ~113.6 | ~4.9 | ~0.827 |
| Original $ -target LGBM+XGB blend | ~160.6 | ~7.3 | ~0.814 |

Why trees: freight pricing has non-linear equipment premiums, lane effects, and seasonal market interactions. Predicting $/mile stabilizes the target across short vs long hauls.

## 7. December chart

For the fixed Lexington → Fort Wayne / 360 mi / Dry Van / 32,000 lb lane:

1. Attach city coordinates from training
2. Fill daily `market_index` from validation-window daily medians
3. Use lane quote-signal prior when quote is absent
4. Run the same blended RPM model

Chart: `scorer_results/candidate_december.png` (produced by `score.py`).

## 8. Deliverables in this repo

- `validation_predictions.csv` — 12,000 rows, `load_id,predicted_rate`
- `december_chart_predictions.csv` — 31 daily predictions
- `scorer_results/candidate_december.png`
- Reproducible code under `src/` via `python run_pipeline.py`
