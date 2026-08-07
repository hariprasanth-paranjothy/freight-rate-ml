"""Generate validation and December chart predictions using the best RPM blend."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import build_matrix, impute_and_enrich
from src.train import predict_blend


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
VAL_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation-predictions-template.csv"
DEC_PATH = DATA_DIR / "december-chart-inputs.csv"
OUT_VAL = ROOT / "validation_predictions.csv"
OUT_DEC = ROOT / "december_chart_predictions.csv"


def enrich_december_with_val_market(dec: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    out = dec.copy()
    val = val.copy()
    val["date"] = pd.to_datetime(val["date"])
    daily = val.groupby("date")["market_index"].median()
    out["date"] = pd.to_datetime(out["date"])
    out["market_index"] = out["date"].map(daily)
    return out


def predict_frame(df: pd.DataFrame, artifact: dict) -> np.ndarray:
    state = artifact["state"]
    fe = impute_and_enrich(df, state)
    X = build_matrix(fe)
    distance = fe["distance"].to_numpy(dtype=float).clip(min=1.0)
    return predict_blend(artifact, X, distance)


def main() -> None:
    artifact = joblib.load(MODELS_DIR / "freight_rate_model.joblib")

    val = pd.read_csv(VAL_PATH)
    template = pd.read_csv(TEMPLATE_PATH)
    print(f"Validation rows: {len(val):,}")

    val_pred = predict_frame(val, artifact)
    submission = template.copy()
    pred_map = dict(zip(val["load_id"].astype(str), val_pred))
    submission["predicted_rate"] = submission["load_id"].astype(str).map(pred_map)
    if submission["predicted_rate"].isna().any():
        missing = int(submission["predicted_rate"].isna().sum())
        raise SystemExit(f"Missing predictions for {missing} load_ids")
    submission.to_csv(OUT_VAL, index=False)
    print(f"Wrote {OUT_VAL}")
    print(submission["predicted_rate"].describe())

    dec = pd.read_csv(DEC_PATH)
    dec_enriched = enrich_december_with_val_market(dec, val)
    dec_pred = predict_frame(dec_enriched, artifact)
    dec_out = dec.copy()
    dec_out["predicted_rate"] = dec_pred
    dec_out = dec_out[
        ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    ]
    dec_out.to_csv(OUT_DEC, index=False)
    print(f"Wrote {OUT_DEC}")
    print(dec_out[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
