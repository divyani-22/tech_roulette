"""Non-ML naive baselines module for EV load benchmarking.

Establishes fundamental heuristic standards (persistence & moving averages)
against which ML regression models must prove their value.
"""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Constants
BASELINE_A_NAME = "Baseline A (Same Hour Yesterday)"
BASELINE_B_NAME = "Baseline B (3-Hour Rolling Mean)"


def predict_naive_yesterday(df: pd.DataFrame) -> np.ndarray:
    """Baseline A: Predicts tomorrow's load using the identical hour from yesterday (lag_24)."""
    if "lag_24" not in df.columns:
        raise ValueError("Feature 'lag_24' must be present for Baseline A.")
    # In case of missing sensor values, fallback to lag_1
    preds = df["lag_24"].fillna(df.get("lag_1", 0.0)).to_numpy()
    return preds


def predict_rolling_mean(df: pd.DataFrame) -> np.ndarray:
    """Baseline B: Predicts next slot using moving average of previous 3 hours."""
    if "rolling_mean_3" not in df.columns:
        raise ValueError("Feature 'rolling_mean_3' must be present for Baseline B.")
    preds = df["rolling_mean_3"].fillna(df.get("lag_1", 0.0)).to_numpy()
    return preds


def evaluate_baseline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_name: str,
) -> dict[str, Any]:
    """Computes MAE and RMSE in kW for a given baseline series."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    return {
        "model": baseline_name,
        "mae_kw": round(float(mae), 2),
        "rmse_kw": round(float(rmse), 2),
        "predictions": y_pred,
    }


def compute_all_baselines(
    df: pd.DataFrame,
    target_col: str = "target_next_kw",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluates both naive baselines on the provided evaluation dataset."""
    y_true = df[target_col].to_numpy()

    preds_a = predict_naive_yesterday(df)
    eval_a = evaluate_baseline(y_true, preds_a, BASELINE_A_NAME)

    preds_b = predict_rolling_mean(df)
    eval_b = evaluate_baseline(y_true, preds_b, BASELINE_B_NAME)

    return eval_a, eval_b
