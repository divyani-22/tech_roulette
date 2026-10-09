"""Evaluation module calculating MAE, RMSE, and benchmark improvements.

Explicitly rejects MAPE due to numerical instability and infinite distortion
when night-time EV charging demand approaches 0 kW.
"""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Constants
DECIMAL_PLACES = 2


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculates MAE and RMSE in kW for regression predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    return {
        "mae_kw": round(float(mae), DECIMAL_PLACES),
        "rmse_kw": round(float(rmse), DECIMAL_PLACES),
    }


def compute_relative_improvement(base_metric: float, model_metric: float) -> float:
    """Computes percentage error reduction relative to the baseline."""
    if base_metric == 0:
        return 0.0
    improvement = ((base_metric - model_metric) / base_metric) * 100.0
    return round(float(improvement), DECIMAL_PLACES)


def build_comparison_summary(
    eval_baseline_a: dict[str, Any],
    eval_baseline_b: dict[str, Any],
    eval_rf: dict[str, Any],
    eval_gbm: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Consolidates all evaluated models into a single comparative leaderboard."""
    benchmark_mae = eval_baseline_a["mae_kw"]
    rows = [
        {
            "Model": eval_baseline_a["model"],
            "MAE (kW)": eval_baseline_a["mae_kw"],
            "RMSE (kW)": eval_baseline_a["rmse_kw"],
            "Improvement vs Base A (%)": 0.0,
        },
        {
            "Model": eval_baseline_b["model"],
            "MAE (kW)": eval_baseline_b["mae_kw"],
            "RMSE (kW)": eval_baseline_b["rmse_kw"],
            "Improvement vs Base A (%)": compute_relative_improvement(benchmark_mae, eval_baseline_b["mae_kw"]),
        },
        {
            "Model": "Random Forest Regressor",
            "MAE (kW)": eval_rf["mae_kw"],
            "RMSE (kW)": eval_rf["rmse_kw"],
            "Improvement vs Base A (%)": compute_relative_improvement(benchmark_mae, eval_rf["mae_kw"]),
        },
    ]

    if eval_gbm is not None:
        rows.append({
            "Model": "Gradient Boosting Regressor",
            "MAE (kW)": eval_gbm["mae_kw"],
            "RMSE (kW)": eval_gbm["rmse_kw"],
            "Improvement vs Base A (%)": compute_relative_improvement(benchmark_mae, eval_gbm["mae_kw"]),
        })

    return pd.DataFrame(rows)


def build_actual_vs_predicted_df(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Random Forest",
) -> pd.DataFrame:
    """Prepares structured test set actuals, predictions, and residuals for plotting."""
    plot_df = pd.DataFrame({
        "timestamp": test_df["timestamp"],
        "station_id": test_df.get("station_id", "Unknown"),
        "actual_kw": np.round(y_true, DECIMAL_PLACES),
        "predicted_kw": np.round(y_pred, DECIMAL_PLACES),
        "residual_kw": np.round(y_pred - y_true, DECIMAL_PLACES),
        "model": model_name,
    })
    return plot_df.sort_values(by="timestamp").reset_index(drop=True)
