"""Data cleaning and feature engineering for freight rate prediction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


EQUIPMENT_ORDER = ["Dry Van", "Flatbed", "Reefer"]


@dataclass
class FeatureState:
    """Fitted statistics used to transform new data without leakage."""

    weight_median_by_equipment: dict[str, float] = field(default_factory=dict)
    global_weight_median: float = 31000.0
    market_index_by_date: dict[pd.Timestamp, float] = field(default_factory=dict)
    global_market_index_median: float = 1.0
    market_index_by_month: dict[int, float] = field(default_factory=dict)
    city_coords: dict[str, tuple[float, float]] = field(default_factory=dict)
    lane_rpm_stats: pd.DataFrame | None = None
    pickup_rpm_stats: pd.DataFrame | None = None
    delivery_rpm_stats: pd.DataFrame | None = None
    equipment_rpm: dict[str, float] = field(default_factory=dict)
    global_rpm: float = 2.15
    quote_signal_median: float = 2.05
    lane_quote_stats: dict[str, float] = field(default_factory=dict)


def _haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in miles."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 3958.8 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _bearing_degrees(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Initial bearing from pickup to delivery (degrees)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Fix obvious data-quality issues before feature creation."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    # Negative weights appear to be sign-flip errors (magnitudes look normal).
    if "weight" in out.columns:
        out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
        out["weight_was_negative"] = (out["weight"] < 0).astype(np.int8)
        out["weight"] = out["weight"].abs()

    for col in ["distance", "market_index", "quote_signal", "posted_rate"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def fit_feature_state(train: pd.DataFrame) -> FeatureState:
    """Learn imputation and target-encoding stats from labeled training rows only."""
    df = clean_raw(train)
    state = FeatureState()

    # Weight medians by equipment
    state.global_weight_median = float(df["weight"].median())
    state.weight_median_by_equipment = (
        df.groupby("equipment")["weight"].median().astype(float).to_dict()
    )

    # Market index: mostly a daily factor
    daily_mi = df.groupby("date")["market_index"].median()
    state.market_index_by_date = {pd.Timestamp(k): float(v) for k, v in daily_mi.items()}
    state.global_market_index_median = float(df["market_index"].median())
    state.market_index_by_month = (
        df.groupby(df["date"].dt.month)["market_index"].median().astype(float).to_dict()
    )

    # City coordinates (mode / first seen)
    for city_col, lat_col, lon_col in [
        ("pickup", "pickup_lat", "pickup_lon"),
        ("delivery", "delivery_lat", "delivery_lon"),
    ]:
        coords = (
            df.groupby(city_col)[[lat_col, lon_col]]
            .median()
            .dropna()
        )
        for city, row in coords.iterrows():
            state.city_coords[str(city)] = (float(row[lat_col]), float(row[lon_col]))

    # Target encodings on rate-per-mile (more stable than raw rate across distances)
    rpm = df["posted_rate"] / df["distance"].clip(lower=1.0)
    df = df.assign(_rpm=rpm, _lane=df["pickup"].astype(str) + "->" + df["delivery"].astype(str))

    state.global_rpm = float(rpm.median())
    state.equipment_rpm = df.groupby("equipment")["_rpm"].median().astype(float).to_dict()

    def _agg(group_col: str) -> pd.DataFrame:
        g = df.groupby(group_col)["_rpm"].agg(["mean", "median", "count"])
        g.columns = ["rpm_mean", "rpm_median", "rpm_count"]
        return g

    state.lane_rpm_stats = _agg("_lane")
    state.pickup_rpm_stats = _agg("pickup")
    state.delivery_rpm_stats = _agg("delivery")

    state.quote_signal_median = float(df["quote_signal"].median())
    lane_quote = df.groupby("_lane")["quote_signal"].median()
    state.lane_quote_stats = lane_quote.astype(float).to_dict()

    return state


def _smoothed_lookup(
    keys: pd.Series,
    stats: pd.DataFrame | None,
    global_value: float,
    min_count: int = 5,
    strength: float = 10.0,
) -> pd.Series:
    """Bayesian-smoothed mean toward the global prior."""
    if stats is None:
        return pd.Series(global_value, index=keys.index)

    joined = keys.map(stats["rpm_mean"]).astype(float)
    counts = keys.map(stats["rpm_count"]).fillna(0).astype(float)
    # Prefer median for sparse lanes
    medians = keys.map(stats["rpm_median"]).astype(float)

    base = joined.where(counts >= min_count, medians)
    w = counts / (counts + strength)
    return (w * base.fillna(global_value) + (1.0 - w) * global_value).fillna(global_value)


def impute_and_enrich(df: pd.DataFrame, state: FeatureState, for_december: bool = False) -> pd.DataFrame:
    """Impute missing fields and attach engineered features."""
    out = clean_raw(df)

    # Fill coordinates from city map when missing (December chart case)
    if "pickup_lat" not in out.columns:
        out["pickup_lat"] = np.nan
        out["pickup_lon"] = np.nan
        out["delivery_lat"] = np.nan
        out["delivery_lon"] = np.nan

    for city_col, lat_col, lon_col in [
        ("pickup", "pickup_lat", "pickup_lon"),
        ("delivery", "delivery_lat", "delivery_lon"),
    ]:
        missing = out[lat_col].isna() | out[lon_col].isna()
        if missing.any():
            coords = out.loc[missing, city_col].map(state.city_coords)
            out.loc[missing, lat_col] = coords.map(lambda x: x[0] if isinstance(x, tuple) else np.nan)
            out.loc[missing, lon_col] = coords.map(lambda x: x[1] if isinstance(x, tuple) else np.nan)

    # Weight imputation
    if "weight_was_negative" not in out.columns:
        out["weight_was_negative"] = 0
    eq_med = out["equipment"].map(state.weight_median_by_equipment)
    out["weight_missing"] = out["weight"].isna().astype(np.int8)
    out["weight"] = out["weight"].fillna(eq_med).fillna(state.global_weight_median)

    # Market index imputation
    if "market_index" not in out.columns:
        out["market_index"] = np.nan
    out["market_index_missing"] = out["market_index"].isna().astype(np.int8)
    date_mi = out["date"].map(lambda d: state.market_index_by_date.get(pd.Timestamp(d), np.nan))
    month_mi = out["date"].dt.month.map(state.market_index_by_month)
    out["market_index"] = (
        out["market_index"]
        .fillna(date_mi)
        .fillna(month_mi)
        .fillna(state.global_market_index_median)
    )

    # Quote signal (December has none)
    if "quote_signal" not in out.columns:
        out["quote_signal"] = np.nan
    out["quote_signal_missing"] = out["quote_signal"].isna().astype(np.int8)
    lane = out["pickup"].astype(str) + "->" + out["delivery"].astype(str)
    lane_q = lane.map(state.lane_quote_stats)
    out["quote_signal"] = out["quote_signal"].fillna(lane_q).fillna(state.quote_signal_median)

    # Core geometry / route features
    out["lane"] = lane
    out["delta_lat"] = out["delivery_lat"] - out["pickup_lat"]
    out["delta_lon"] = out["delivery_lon"] - out["pickup_lon"]
    out["abs_delta_lat"] = out["delta_lat"].abs()
    out["abs_delta_lon"] = out["delta_lon"].abs()
    out["haversine_miles"] = _haversine_miles(
        out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"]
    )
    out["bearing"] = _bearing_degrees(
        out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"]
    )
    out["distance_vs_haversine"] = out["distance"] / out["haversine_miles"].clip(lower=1.0)
    out["log_distance"] = np.log1p(out["distance"])
    out["sqrt_distance"] = np.sqrt(out["distance"].clip(lower=0))

    # Temporal
    out["month"] = out["date"].dt.month
    out["day_of_week"] = out["date"].dt.dayofweek
    out["day_of_year"] = out["date"].dt.dayofyear
    out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(np.int8)
    out["day_of_month"] = out["date"].dt.day
    out["is_month_start"] = (out["day_of_month"] <= 3).astype(np.int8)
    out["is_month_end"] = (out["day_of_month"] >= 28).astype(np.int8)
    # Cyclical encodings
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365)

    # Interactions aligned with freight pricing intuition
    out["distance_x_market"] = out["distance"] * out["market_index"]
    out["distance_x_quote"] = out["distance"] * out["quote_signal"]
    out["market_x_quote"] = out["market_index"] * out["quote_signal"]
    out["expected_base_rate"] = out["distance"] * out["quote_signal"] * out["market_index"]
    out["weight_per_mile"] = out["weight"] / out["distance"].clip(lower=1.0)
    out["log_weight"] = np.log1p(out["weight"])
    out["heavy_load"] = (out["weight"] >= 40000).astype(np.int8)

    # Target encodings (fit on train only)
    out["lane_rpm"] = _smoothed_lookup(out["lane"], state.lane_rpm_stats, state.global_rpm)
    out["pickup_rpm"] = _smoothed_lookup(out["pickup"], state.pickup_rpm_stats, state.global_rpm)
    out["delivery_rpm"] = _smoothed_lookup(
        out["delivery"], state.delivery_rpm_stats, state.global_rpm
    )
    out["equipment_rpm"] = out["equipment"].map(state.equipment_rpm).fillna(state.global_rpm)
    out["lane_prior_rate"] = out["lane_rpm"] * out["distance"]

    # Equipment ordinal + one-hot friendly category code
    eq_map = {e: i for i, e in enumerate(EQUIPMENT_ORDER)}
    out["equipment_code"] = out["equipment"].map(eq_map).fillna(-1).astype(int)

    if for_december:
        # Flag so model can learn that Dec rows used imputed market/quote if needed
        pass

    return out


FEATURE_COLUMNS = [
    # raw / imputed numerics
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    # quality flags
    "weight_missing",
    "weight_was_negative",
    "market_index_missing",
    "quote_signal_missing",
    # geometry
    "delta_lat",
    "delta_lon",
    "abs_delta_lat",
    "abs_delta_lon",
    "haversine_miles",
    "bearing",
    "distance_vs_haversine",
    "log_distance",
    "sqrt_distance",
    # temporal
    "month",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "is_weekend",
    "day_of_month",
    "is_month_start",
    "is_month_end",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    # interactions
    "distance_x_market",
    "distance_x_quote",
    "market_x_quote",
    "expected_base_rate",
    "weight_per_mile",
    "log_weight",
    "heavy_load",
    # encodings
    "lane_rpm",
    "pickup_rpm",
    "delivery_rpm",
    "equipment_rpm",
    "lane_prior_rate",
    "equipment_code",
]

CATEGORICAL_COLUMNS = ["equipment", "pickup", "delivery"]


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return model-ready feature matrix."""
    cols = FEATURE_COLUMNS + CATEGORICAL_COLUMNS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing engineered columns: {missing}")
    X = df[cols].copy()
    for c in CATEGORICAL_COLUMNS:
        X[c] = X[c].astype("category")
    return X
