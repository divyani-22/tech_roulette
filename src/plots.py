"""Plotly visualization suite for VoltCap.

Generates custom dark 'energy control room' interactive charts
with electric cyan (#22D3EE), lime (#A3E635), amber (#F59E0B), and red (#EF4444).
"""

from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Color constants
COLOR_BG = "#131A2E"
COLOR_PAPER = "#0B1020"
COLOR_CYAN = "#22D3EE"
COLOR_LIME = "#A3E635"
COLOR_AMBER = "#F59E0B"
COLOR_RED = "#EF4444"
COLOR_TEXT = "#E5E7EB"
COLOR_MUTED = "#94A3B8"


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Applies unified energy control room dark aesthetics to Plotly charts."""
    fig.update_layout(
        paper_bgcolor=COLOR_PAPER,
        plot_bgcolor=COLOR_BG,
        font=dict(family="Inter, sans-serif", color=COLOR_TEXT, size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.1)"),
    )
    return fig


def plot_demand_timeseries(df: pd.DataFrame, station_id: str) -> go.Figure:
    """Renders demand time series with overlaid outlier markers."""
    st_df = df[df["station_id"] == station_id].sort_values("timestamp")
    fig = go.Figure()

    # Normal demand line
    fig.add_trace(go.Scatter(
        x=st_df["timestamp"],
        y=st_df["demand_kw"],
        mode="lines",
        name="Clean Demand (kW)",
        line=dict(color=COLOR_CYAN, width=2),
    ))

    # Highlight capped outliers if flagged
    if "is_outlier" in st_df.columns and st_df["is_outlier"].any():
        outliers = st_df[st_df["is_outlier"]]
        fig.add_trace(go.Scatter(
            x=outliers["timestamp"],
            y=outliers.get("demand_kw_raw", outliers["demand_kw"]),
            mode="markers",
            name="Capped Outlier Spike",
            marker=dict(color=COLOR_RED, size=8, symbol="diamond"),
        ))

    fig.update_layout(
        title=f"Hourly Demand Profile — {station_id}",
        xaxis_title="Timestamp",
        yaxis_title="Power Demand (kW)",
    )
    return apply_dark_theme(fig)


def plot_model_comparison_bar(summary_df: pd.DataFrame) -> go.Figure:
    """Renders MAE and RMSE performance comparison bars across baselines and models."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=summary_df["Model"],
        y=summary_df["MAE (kW)"],
        name="MAE (kW)",
        marker_color=COLOR_CYAN,
    ))
    fig.add_trace(go.Bar(
        x=summary_df["Model"],
        y=summary_df["RMSE (kW)"],
        name="RMSE (kW)",
        marker_color=COLOR_AMBER,
    ))
    fig.update_layout(
        title="Error Metrics: Naive Baselines vs Trained ML Models (Lower is Better)",
        barmode="group",
        yaxis_title="Error (kW)",
    )
    return apply_dark_theme(fig)


def plot_actual_vs_predicted(eval_df: pd.DataFrame, n_points: int = 120) -> go.Figure:
    """Plots chronological test set actuals versus ML forecasts."""
    sample = eval_df.tail(n_points).copy()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sample["timestamp"],
        y=sample["actual_kw"],
        mode="lines",
        name="Actual Demand (kW)",
        line=dict(color="#94A3B8", width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=sample["timestamp"],
        y=sample["predicted_kw"],
        mode="lines",
        name="VoltCap ML Forecast (kW)",
        line=dict(color=COLOR_CYAN, width=2.5, dash="solid"),
    ))
    fig.update_layout(
        title=f"Test Set: Actual vs Predicted Demand (Last {n_points} Hours)",
        xaxis_title="Timestamp",
        yaxis_title="Power Demand (kW)",
    )
    return apply_dark_theme(fig)


def plot_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    """Renders horizontal bar chart of normalized Gini feature importances."""
    sorted_df = importance_df.sort_values(by="importance", ascending=True)
    fig = go.Figure(go.Bar(
        x=sorted_df["importance"],
        y=sorted_df["feature"],
        orientation="h",
        marker=dict(color=COLOR_CYAN),
    ))
    fig.update_layout(
        title="Random Forest Feature Importance (MDI)",
        xaxis_title="Relative Importance Weight",
        yaxis_title="Feature",
    )
    return apply_dark_theme(fig)


def plot_prediction_gauge(pred_kw: float, cap_kw: float, low_kw: float, high_kw: float) -> go.Figure:
    """Displays forecast load vs transformer power cap on an interactive gauge."""
    ratio = pred_kw / max(cap_kw, 0.001)
    bar_color = COLOR_CYAN if ratio < 0.6 else (COLOR_LIME if ratio < 0.85 else (COLOR_AMBER if ratio < 1.0 else COLOR_RED))

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pred_kw,
        delta={"reference": cap_kw, "increasing": {"color": COLOR_RED}, "decreasing": {"color": COLOR_LIME}},
        title={"text": f"Next-Hour Forecast: {pred_kw:.1f} kW<br><span style='font-size:13px;color:#94A3B8'>Cap: {cap_kw:.1f} kW (80% CI: [{low_kw:.1f}, {high_kw:.1f}])</span>"},
        gauge={
            "axis": {"range": [0, max(cap_kw * 1.35, pred_kw * 1.2)]},
            "bar": {"color": bar_color},
            "steps": [
                {"range": [0, cap_kw * 0.6], "color": "rgba(34, 211, 238, 0.15)"},
                {"range": [cap_kw * 0.6, cap_kw * 0.85], "color": "rgba(163, 230, 53, 0.15)"},
                {"range": [cap_kw * 0.85, cap_kw], "color": "rgba(245, 158, 11, 0.15)"},
                {"range": [cap_kw, max(cap_kw * 1.35, pred_kw * 1.2)], "color": "rgba(239, 68, 68, 0.2)"},
            ],
            "threshold": {"line": {"color": COLOR_RED, "width": 4}, "thickness": 0.85, "value": cap_kw},
        },
    ))
    return apply_dark_theme(fig)


def plot_ev_allocations(sched_df: pd.DataFrame, cap_kw: float) -> go.Figure:
    """Draws per-EV allocated power stacked bar with explicit red dashed cap line."""
    fig = go.Figure()
    # Color map for decisions
    color_map = {"Charge Now": COLOR_LIME, "Throttle": COLOR_AMBER, "Defer": COLOR_RED}

    for decision, color in color_map.items():
        subset = sched_df[sched_df["decision"] == decision]
        if not subset.empty:
            fig.add_trace(go.Bar(
                x=subset["ev_id"],
                y=subset["allocated_kw"],
                name=decision,
                marker_color=color,
                text=[f"{v:.1f} kW" for v in subset["allocated_kw"]],
                textposition="auto",
            ))

    # Add red dashed cap reference line
    fig.add_hline(
        y=cap_kw,
        line_dash="dash",
        line_color=COLOR_RED,
        line_width=3,
        annotation_text=f"HARD POWER CAP: {cap_kw:.1f} kW",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Per-EV Power Allocation (Greedy Earliest-Deadline-First)",
        xaxis_title="Vehicle Identifier (Sorted by Urgency)",
        yaxis_title="Allocated Power (kW)",
        barmode="group",
    )
    return apply_dark_theme(fig)


def plot_peak_comparison(impact_dict: dict[str, Any]) -> go.Figure:
    """Compares uncontrolled peak surge demand vs VoltCap-regulated peak."""
    fig = go.Figure()
    categories = ["Uncontrolled Surge", "VoltCap Scheduled"]
    values = [impact_dict["uncontrolled_peak_kw"], impact_dict["scheduled_peak_kw"]]
    colors = [COLOR_RED, COLOR_LIME]

    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f} kW" for v in values],
        textposition="auto",
    ))
    fig.update_layout(
        title="Peak Shaving: Uncontrolled Plug-in vs Cap-Safe Schedule",
        yaxis_title="Total Slot Load (kW)",
    )
    return apply_dark_theme(fig)
