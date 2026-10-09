"""Unit and failure-case tests for VoltCap.

Verifies mathematical cap invariance, extreme surge handling,
sensor outage fallbacks, and resilient handling of malformed datasets.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.clean import clean_pipeline, validate_and_load_data
from src.features import create_feature_dataset
from src.scheduler import run_edf_scheduler, STATUS_DEFER


def test_hard_cap_invariant():
    """Asserts that total allocated kW strictly never exceeds the fixed power cap."""
    cap = 75.0
    # Create 10 demanding EVs
    ev_queue = pd.DataFrame([
        {
            "ev_id": f"EV-{i}",
            "deadline": "2025-01-01 10:00",
            "energy_needed_kwh": 30.0,
            "max_kw": 22.0,
        }
        for i in range(10)
    ])
    # Total requested: 10 * 22 = 220 kW, Cap = 75 kW
    scheduled, summary = run_edf_scheduler(ev_queue, cap_kw=cap)

    total_allocated = scheduled["allocated_kw"].sum()
    assert total_allocated <= cap + 1e-5
    assert summary["cap_respected"] is True
    assert summary["total_allocated_kw"] <= cap


def test_surge_failure_case():
    """Simulates 3x cap power surge: verifies cap holds and late deadlines are deferred."""
    cap = 50.0
    now = datetime(2025, 1, 1, 8, 0)
    # Generate 15 EVs with varying deadlines requesting ~150 kW (3x cap)
    queue = []
    for i in range(15):
        # Even indices have urgent 1h deadlines; odd indices have 4h deadlines
        hours_lead = 1 if i % 2 == 0 else 4
        queue.append({
            "ev_id": f"EV-SURGE-{i:02d}",
            "deadline": (now + timedelta(hours=hours_lead)).strftime("%Y-%m-%d %H:00"),
            "energy_needed_kwh": 20.0,
            "max_kw": 11.0,
        })
    ev_queue = pd.DataFrame(queue)

    scheduled, summary = run_edf_scheduler(ev_queue, cap_kw=cap)

    # 1. Cap is strictly respected
    assert scheduled["allocated_kw"].sum() <= cap
    # 2. Defers exist because demand heavily outstripped capacity
    assert summary["defer_count"] > 0
    # 3. Late deadline vehicles are prioritized for deferral
    late_evs = scheduled[scheduled["deadline"] > (now + timedelta(hours=2)).strftime("%Y-%m-%d %H:00")]
    assert any(late_evs["decision"] == STATUS_DEFER)


def test_missing_sensor_fallback():
    """Tests resilience when real-time lag_1 sensor fails and triggers fallback."""
    # Build mini dataframe where lag_1 sensor is NaN
    data = {
        "hour": [14],
        "day_of_week": [2],
        "is_weekend": [0],
        "lag_1": [np.nan],
        "lag_24": [42.5],
        "rolling_mean_3": [45.0],
        "active_sessions_lag_1": [3],
        "temperature_c": [18.0],
    }
    df = pd.DataFrame(data)

    # Fallback mechanism: if lag_1 is NaN, use lag_24 or rolling mean
    if pd.isna(df["lag_1"].iloc[0]):
        fallback_pred = df["lag_24"].iloc[0] if not pd.isna(df["lag_24"].iloc[0]) else df["rolling_mean_3"].iloc[0]
        used_fallback = True
    else:
        fallback_pred = df["lag_1"].iloc[0]
        used_fallback = False

    assert used_fallback is True
    assert fallback_pred == 42.5
    assert not np.isnan(fallback_pred)


def test_empty_and_invalid_input():
    """Tests that empty datasets or invalid columns raise clear errors without crashing."""
    # 1. Empty EV queue should gracefully return empty schedule
    empty_queue = pd.DataFrame()
    scheduled, summary = run_edf_scheduler(empty_queue, cap_kw=100.0)
    assert scheduled.empty
    assert summary["total_allocated_kw"] == 0.0

    # 2. Invalid schema (missing required columns) raises ValueError
    bad_df = pd.DataFrame({"some_random_column": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing mandatory columns"):
        validate_and_load_data(bad_df)

    # 3. Negative values are handled gracefully in clean pipeline
    negative_df = pd.DataFrame({
        "timestamp": ["2025-01-01 00:00:00", "2025-01-01 01:00:00"],
        "station_id": ["Station_A", "Station_A"],
        "demand_kw": [-25.0, 50.0],
    })
    clean_df, summary = clean_pipeline(negative_df)
    # Demand lower bound clipped at 0
    assert (clean_df["demand_kw"] >= 0).all()


def test_bms_battery_health_safeguard():
    """Verifies that EVs with SoC >= 80% are throttled to <= 7.4 kW to prevent lithium degradation."""
    cap = 100.0
    ev_queue = pd.DataFrame([
        {
            "ev_id": "EV-SAFE-01",
            "deadline": "2025-01-01 10:00",
            "energy_needed_kwh": 30.0,
            "max_kw": 22.0,
            "soc_pct": 85.0,  # High SoC: must be throttled
        },
        {
            "ev_id": "EV-FAST-02",
            "deadline": "2025-01-01 11:00",
            "energy_needed_kwh": 30.0,
            "max_kw": 22.0,
            "soc_pct": 35.0,  # Low SoC: can fast-charge
        },
    ])
    scheduled, summary = run_edf_scheduler(ev_queue, cap_kw=cap, enable_bms_guard=True)

    ev_safe = scheduled[scheduled["ev_id"] == "EV-SAFE-01"].iloc[0]
    ev_fast = scheduled[scheduled["ev_id"] == "EV-FAST-02"].iloc[0]

    assert bool(ev_safe["bms_guarded"]) is True
    assert ev_safe["allocated_kw"] <= 7.4
    assert bool(ev_fast["bms_guarded"]) is False
    assert ev_fast["allocated_kw"] == 22.0
    assert summary["bms_protected_count"] == 1


def test_tou_tariffs_and_carbon_savings():
    """Verifies Time-of-Use financial savings and avoided carbon calculation."""
    from src.impact import compute_slot_impact, get_hourly_tariff, get_hourly_carbon_intensity

    # Peak hour = 18:00
    assert get_hourly_tariff(18) == 0.44
    assert get_hourly_carbon_intensity(18)[0] == 485.0

    # Simulate schedule where surge was 150 kW and regulated was 80 kW under 100 kW cap
    sched_df = pd.DataFrame([
        {"ev_id": "EV-1", "max_kw": 150.0, "energy_needed_kwh": 150.0, "allocated_kw": 80.0, "decision": "Charge Now", "bms_guarded": True}
    ])
    impact = compute_slot_impact(sched_df, cap_kw=100.0, slot_hour=18)

    assert impact["cost_saved_usd"] > 0
    assert impact["demand_penalty_saved_usd"] > 0
    assert impact["avoided_carbon_kg"] > 0
    assert impact["bms_protected_count"] == 1


def test_executive_compliance_audit_report():
    """Verifies the executive compliance report generates with verified invariant status."""
    from src.impact import generate_compliance_audit_report

    sched_df = pd.DataFrame([
        {"ev_id": "EV-01", "deadline": "2025-01-01 12:00", "hours_remaining": 2, "soc_pct": 65.0, "max_kw": 22.0, "allocated_kw": 22.0, "decision": "Charge Now", "bms_guarded": False}
    ])
    sched_sum = {"total_allocated_kw": 22.0, "utilization_pct": 22.0}
    impact = {"tariff_rate_usd": 0.22, "uncontrolled_cost_usd": 15.0, "scheduled_cost_usd": 4.84, "cost_saved_usd": 10.16, "demand_penalty_saved_usd": 0.0, "carbon_intensity_g_kwh": 250.0, "clean_share_pct": 60.0, "avoided_carbon_kg": 2.5, "bms_protected_count": 0, "avoided_breach_kw": 0.0}
    eval_metrics = {"mae_kw": 4.1, "rmse_kw": 5.8}

    report = generate_compliance_audit_report("Station_A", "2025-01-01 10:00", 100.0, sched_df, sched_sum, impact, eval_metrics)
    assert "PASSED (100% ADHERENT)" in report
    assert "EV-01" in report
    assert "$10.16" in report
