"""Train the best holdout blend: RPM targets with LGBM + CatBoost + ExtraTrees."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import (
    CATEGORICAL_COLUMNS,
    build_matrix,
    fit_feature_state,
    impute_and_enrich,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

BLEND_WEIGHTS = {
    "lgbm": 0.50,
    "catboost": 0.30,
    "extra_trees": 0.20,
}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.clip(y_pred, 1.0, None)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1.0, None))) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }


def time_split(df: pd.DataFrame, holdout_start: str = "2025-09-01"):
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    cutoff = pd.Timestamp(holdout_start)
    return out[out["date"] < cutoff].copy(), out[out["date"] >= cutoff].copy()


def encode_numeric(X_train: pd.DataFrame, X_other: pd.DataFrame):
    Xtr, Xot = X_train.copy(), X_other.copy()
    maps: dict[str, dict] = {}
    for col in CATEGORICAL_COLUMNS:
        vals = Xtr[col].astype(str)
        mapping = {c: i for i, c in enumerate(sorted(vals.unique()))}
        maps[col] = mapping
        Xtr[col] = vals.map(mapping).fillna(-1).astype(int)
        Xot[col] = Xot[col].astype(str).map(mapping).fillna(-1).astype(int)
    return Xtr.astype(float), Xot.astype(float), maps


def catboost_frames(X_train: pd.DataFrame, X_other: pd.DataFrame):
    Xtr, Xot = X_train.copy(), X_other.copy()
    for c in CATEGORICAL_COLUMNS:
        Xtr[c] = Xtr[c].astype(str)
        Xot[c] = Xot[c].astype(str)
    cat_idx = [Xtr.columns.get_loc(c) for c in CATEGORICAL_COLUMNS]
    return Xtr, Xot, cat_idx


def train_lgbm_rpm(X_train, rpm_train, X_valid, rpm_valid) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        n_estimators=4000,
        learning_rate=0.03,
        num_leaves=127,
        min_child_samples=25,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="mae",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X_train,
        rpm_train,
        eval_set=[(X_valid, rpm_valid)],
        eval_metric="l1",
        categorical_feature=CATEGORICAL_COLUMNS,
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def train_catboost_rpm(X_train, rpm_train, X_valid, rpm_valid, cat_idx) -> CatBoostRegressor:
    model = CatBoostRegressor(
        iterations=3500,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3.0,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=42,
        early_stopping_rounds=150,
        verbose=False,
    )
    model.fit(
        X_train,
        rpm_train,
        eval_set=(X_valid, rpm_valid),
        cat_features=cat_idx,
        use_best_model=True,
    )
    return model


def train_et_rpm(X_train, rpm_train) -> ExtraTreesRegressor:
    model = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=28,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, rpm_train)
    return model


def predict_blend(artifact: dict, X: pd.DataFrame, distance: np.ndarray) -> np.ndarray:
    """Predict posted_rate from engineered matrix + distance."""
    lgb_rpm = artifact["lgbm"].predict(X)

    X_cb = X.copy()
    for c in CATEGORICAL_COLUMNS:
        X_cb[c] = X_cb[c].astype(str)
    cb_rpm = artifact["catboost"].predict(X_cb)

    X_num = X.copy()
    for col in CATEGORICAL_COLUMNS:
        X_num[col] = X_num[col].astype(str).map(artifact["et_cat_maps"][col]).fillna(-1).astype(int)
    et_rpm = artifact["extra_trees"].predict(X_num.astype(float))

    w = artifact["blend_weights"]
    rpm = w["lgbm"] * lgb_rpm + w["catboost"] * cb_rpm + w["extra_trees"] * et_rpm
    return np.clip(rpm * distance, 1.0, None)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    train_path = DATA_DIR / "train-test.csv"
    if not train_path.exists():
        train_path = ROOT / "train-test.csv"
    raw = pd.read_csv(train_path)
    print(f"Loaded train-test: {raw.shape}")

    pre_train, pre_valid = time_split(raw, "2025-09-01")
    print(f"Time split -> train {len(pre_train):,} | holdout {len(pre_valid):,}")

    state = fit_feature_state(pre_train)
    train_fe = impute_and_enrich(pre_train, state)
    valid_fe = impute_and_enrich(pre_valid, state)
    X_train = build_matrix(train_fe)
    X_valid = build_matrix(valid_fe)
    y_train = train_fe["posted_rate"].to_numpy(float)
    y_valid = valid_fe["posted_rate"].to_numpy(float)
    d_train = train_fe["distance"].to_numpy(float).clip(min=1.0)
    d_valid = valid_fe["distance"].to_numpy(float).clip(min=1.0)
    rpm_train = y_train / d_train
    rpm_valid = y_valid / d_valid

    Xtr_cb, Xva_cb, cat_idx = catboost_frames(X_train, X_valid)
    Xtr_num, Xva_num, et_maps = encode_numeric(X_train, X_valid)

    print("Training LightGBM (RPM / MAE)...")
    lgbm = train_lgbm_rpm(X_train, rpm_train, X_valid, rpm_valid)
    print("Training CatBoost (RPM / MAE)...")
    cat = train_catboost_rpm(Xtr_cb, rpm_train, Xva_cb, rpm_valid, cat_idx)
    print("Training ExtraTrees (RPM)...")
    et = train_et_rpm(Xtr_num, rpm_train)

    holdout_pred = predict_blend(
        {
            "lgbm": lgbm,
            "catboost": cat,
            "extra_trees": et,
            "et_cat_maps": et_maps,
            "blend_weights": BLEND_WEIGHTS,
        },
        X_valid,
        d_valid,
    )
    holdout_metrics = metrics(y_valid, holdout_pred)
    print("Holdout blend metrics:", holdout_metrics)

    # Retrain on full labeled data with chronological early-stopping monitor
    print("Retraining on full labeled data...")
    full_state = fit_feature_state(raw)
    full_fe = impute_and_enrich(raw, full_state).sort_values("date").reset_index(drop=True)
    X_full = build_matrix(full_fe)
    y_full = full_fe["posted_rate"].to_numpy(float)
    d_full = full_fe["distance"].to_numpy(float).clip(min=1.0)
    rpm_full = y_full / d_full
    cut = int(len(full_fe) * 0.85)
    X_tr, X_mon = X_full.iloc[:cut], X_full.iloc[cut:]
    rpm_tr, rpm_mon = rpm_full[:cut], rpm_full[cut:]

    Xtr_cb, Xmon_cb, cat_idx = catboost_frames(X_tr, X_mon)
    Xtr_num, _, et_maps = encode_numeric(X_tr, X_mon)

    final_lgbm = train_lgbm_rpm(X_tr, rpm_tr, X_mon, rpm_mon)
    final_cat = train_catboost_rpm(Xtr_cb, rpm_tr, Xmon_cb, rpm_mon, cat_idx)
    final_et = train_et_rpm(Xtr_num, rpm_tr)

    artifact = {
        "state": full_state,
        "lgbm": final_lgbm,
        "catboost": final_cat,
        "extra_trees": final_et,
        "et_cat_maps": et_maps,
        "blend_weights": BLEND_WEIGHTS,
        "holdout_metrics": holdout_metrics,
        "target": "rate_per_mile_then_times_distance",
    }
    joblib.dump(artifact, MODELS_DIR / "freight_rate_model.joblib")
    with open(OUTPUTS_DIR / "holdout_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"blend": holdout_metrics, "blend_weights": BLEND_WEIGHTS}, f, indent=2)
    print(f"Saved model -> {MODELS_DIR / 'freight_rate_model.joblib'}")


if __name__ == "__main__":
    main()
