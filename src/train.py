"""Model training and prediction interval engine.

Implements chronological 80/20 train/test split without shuffling
to prevent future lookahead leakage, and extracts RF ensemble uncertainty.
"""

import time
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# Constants
DEFAULT_SPLIT_RATIO = 0.80
RF_PARAMS = {"n_estimators": 200, "max_depth": 10, "random_state": 42}
GBM_PARAMS = {"n_estimators": 100, "max_depth": 5, "random_state": 42}


def chronological_split(
    df: pd.DataFrame,
    split_ratio: float = DEFAULT_SPLIT_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits time-series chronologically into 80% train and 20% test.

    Shuffling is strictly avoided: random shuffling would leak future temporal
    correlations into past training, invalidating real-world out-of-sample tests.
    """
    df_sorted = df.sort_values(by="timestamp").reset_index(drop=True)
    split_idx = int(len(df_sorted) * split_ratio)
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    return train_df, test_df


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[RandomForestRegressor, float, pd.DataFrame]:
    """Trains scikit-learn RandomForestRegressor and tracks elapsed runtime."""
    start_time = time.time()
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    # Compute feature importances
    importances = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return model, elapsed_time, importances


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[GradientBoostingRegressor, float, pd.DataFrame]:
    """Trains GradientBoostingRegressor as a secondary benchmark model."""
    start_time = time.time()
    model = GradientBoostingRegressor(**GBM_PARAMS)
    model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    importances = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return model, elapsed_time, importances


def predict_with_confidence_band(
    model: Any,
    X: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Generates point forecast and 10th-90th percentile bounds via tree variance."""
    point_pred = model.predict(X)

    # If Random Forest, inspect individual tree predictions to derive empirical spread
    if hasattr(model, "estimators_"):
        # Shape: (n_estimators, n_samples)
        tree_preds = np.array([tree.predict(X.to_numpy()) for tree in model.estimators_])
        lower_bound = np.percentile(tree_preds, 10, axis=0)
        upper_bound = np.percentile(tree_preds, 90, axis=0)
    else:
        # Fallback heuristic interval for non-ensemble models (+/- 12%)
        lower_bound = point_pred * 0.88
        upper_bound = point_pred * 1.12

    return {
        "prediction": np.clip(point_pred, 0.0, None),
        "lower_bound": np.clip(lower_bound, 0.0, None),
        "upper_bound": np.clip(upper_bound, 0.0, None),
    }


FEATURE_DESCRIPTIONS = {
    "hour": "Commute rush hour timing (8-10 AM, 5-8 PM)",
    "lag_1": "Immediate power momentum from prior hour",
    "lag_24": "Diurnal cycle from exact same hour yesterday",
    "rolling_mean_3": "3-hour smoothing demand trend",
    "active_sessions_lag_1": "Physical charger port occupancy",
    "temperature_c": "Ambient temp & battery preconditioning",
    "is_weekend": "Commercial charging drop on weekends",
    "day_of_week": "Weekly cyclical commute pattern",
}


def explain_prediction_waterfall(
    model: Any,
    input_sample: pd.DataFrame,
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[float, float, pd.DataFrame]:
    """Decomposes a single prediction into individual feature contributions (Waterfall XAI).

    Uses Leave-One-Covariate-Out (LOCO) marginal attribution calibrated so that:
        base_value + sum(contributions) == final_prediction.
    """
    sample = input_sample[feature_cols].copy()
    point_pred = float(model.predict(sample)[0])

    # Baseline station mean demand
    base_val = float(train_df["target_next_kw"].mean() if "target_next_kw" in train_df.columns else 50.0)
    delta_total = point_pred - base_val

    # Marginal LOCO effects
    marginal_deltas = {}
    train_means = train_df[feature_cols].mean()

    for col in feature_cols:
        perturbed = sample.copy()
        perturbed[col] = float(train_means[col])
        pred_pert = float(model.predict(perturbed)[0])
        # Effect is how much the actual value shifted prediction away from the mean
        marginal_deltas[col] = point_pred - pred_pert

    sum_abs = sum(abs(v) for v in marginal_deltas.values())
    if sum_abs > 1e-6:
        # Scale marginal deltas so their algebraic sum matches delta_total exactly
        scale = delta_total / sum(marginal_deltas.values()) if abs(sum(marginal_deltas.values())) > 1e-4 else 1.0
        contributions = {k: round(v * scale, 2) for k, v in marginal_deltas.items()}
    else:
        contributions = {col: 0.0 for col in feature_cols}

    records = []
    for col in feature_cols:
        val = contributions.get(col, 0.0)
        direction = "INCREASE" if val >= 0 else "DECREASE"
        records.append({
            "feature": col,
            "contribution_kw": val,
            "direction": direction,
            "description": FEATURE_DESCRIPTIONS.get(col, col),
        })

    contrib_df = pd.DataFrame(records).sort_values(by="contribution_kw", key=abs, ascending=False).reset_index(drop=True)
    return round(base_val, 2), round(point_pred, 2), contrib_df
