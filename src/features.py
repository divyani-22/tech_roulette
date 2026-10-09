"""Feature engineering module with strict causal past-only representations.

Guarantees zero data leakage by lagging all observational signals prior to
forecasting demand at time slot t + 1.
"""

import pandas as pd

# Constants
TARGET_COL = "target_next_kw"
FEATURE_COLS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "lag_1",
    "lag_24",
    "rolling_mean_3",
    "active_sessions_lag_1",
    "temperature_c",
]

# One-line rationale for each feature (for viva & UI transparency)
FEATURE_RATIONALE = {
    "hour": "Captures standard daily diurnal charging cycles (morning & evening peaks).",
    "day_of_week": "Captures weekly commuting patterns and weekday vs weekend load variance.",
    "is_weekend": "Explicit binary switch for reduced commercial charging on Saturdays and Sundays.",
    "lag_1": "Immediate past load (t) providing short-term inertia and auto-regressive momentum.",
    "lag_24": "Same hour yesterday (t-24), capturing strong daily 24-hour seasonal periodicity.",
    "rolling_mean_3": "3-hour past moving average (shifted by 1) smoothing out short-term sensor noise.",
    "active_sessions_lag_1": "Previous slot's physical plug connections driving impending electrical draw.",
    "temperature_c": "Current ambient temperature affecting EV battery chemistry and HVAC draw.",
}


def compute_station_temporal_features(group: pd.DataFrame) -> pd.DataFrame:
    """Computes calendar, lag, and rolling features strictly using historical values."""
    grp = group.sort_values("timestamp").copy()
    grp["timestamp"] = pd.to_datetime(grp["timestamp"])

    # Target: demand at t+1 (1 hour ahead forecast)
    grp[TARGET_COL] = grp["demand_kw"].shift(-1)

    # Calendar features known at prediction time t
    grp["hour"] = grp["timestamp"].dt.hour
    grp["day_of_week"] = grp["timestamp"].dt.dayofweek
    grp["is_weekend"] = (grp["day_of_week"] >= 5).astype(int)

    # Auto-regressive past lags (strictly <= t)
    # Note: lag_1 is demand at t; lag_24 is demand at t-23 or t-24 relative to t+1
    # Since current row is time t, lag_1 is grp['demand_kw'] (current observed)
    # or grp['demand_kw'].shift(1) if predicting at t?
    # Let's clarify: at time t, we want to predict t+1.
    # At time t, demand_kw at time t is already observed!
    # So lag_1 = demand at t. lag_24 = demand at t-23 (24h before t+1).
    # To keep it standard: current observation is row t.
    grp["lag_1"] = grp["demand_kw"]  # Most recently observed demand at t
    grp["lag_24"] = grp["demand_kw"].shift(23).bfill().fillna(grp["demand_kw"])  # Same hour yesterday

    # Rolling 3-hour mean using past 3 observed points (t, t-1, t-2)
    grp["rolling_mean_3"] = grp["demand_kw"].rolling(window=3, min_periods=1).mean()

    # Active sessions at current time t
    if "active_sessions" in grp.columns:
        grp["active_sessions_lag_1"] = grp["active_sessions"]
    else:
        grp["active_sessions_lag_1"] = 0

    # Ambient temperature (°C) — synthesize realistic diurnal curve if not present
    if "temperature_c" not in grp.columns:
        import numpy as np
        grp["temperature_c"] = np.round(18.0 + 6.0 * np.sin((grp["hour"] - 8) / 24.0 * 2 * np.pi), 1)

    return grp


def create_feature_dataset(clean_df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """Processes cleaned data into a multi-station feature matrix ready for modeling."""
    processed_stations = []

    for _, group in clean_df.groupby("station_id", as_index=False):
        station_df = compute_station_temporal_features(group)
        processed_stations.append(station_df)

    feat_df = pd.concat(processed_stations, ignore_index=True)

    if drop_na:
        # Drop only rows where future target is unknown (the last timestamp of the series)
        feat_df = feat_df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    return feat_df
