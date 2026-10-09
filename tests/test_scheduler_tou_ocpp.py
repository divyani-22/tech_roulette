"""Unit and integration tests for enhanced scheduler, ToU economics,
temperature BMS safeguards, and OCPP 1.6J/2.0.1 profile generator.
"""

from datetime import datetime, timezone
from pathlib import Path
import json
import pandas as pd
import pytest

from voltcap.data.generate_data import generate_ev_queue
from voltcap.src.clean import clean_pipeline
from voltcap.src.features import create_feature_dataset
from voltcap.src.scheduler import (
    compute_risk_level,
    compute_tou_economics,
    generate_ocpp_charging_profiles,
    get_tou_tariff,
    run_edf_scheduler,
    STATUS_CHARGE_NOW,
    STATUS_DEFER,
    STATUS_THROTTLE,
)


@pytest.fixture
def sample_ev_queue():
    ts = pd.Timestamp("2026-10-09 17:00:00")
    return generate_ev_queue(ts, "Station_A", n_evs=8)


def test_tou_tariff_tiers():
    """Verify standard California ToU tariff hours and rates."""
    # On-Peak: 16:00 to 20:59
    on_peak = get_tou_tariff(17)
    assert on_peak["tier"] == "On-Peak"
    assert on_peak["rate_per_kwh"] == 0.46
    assert on_peak["demand_charge_per_kw"] == 18.50

    # Mid-Peak: 12:00
    mid_peak = get_tou_tariff(12)
    assert mid_peak["tier"] == "Mid-Peak"
    assert mid_peak["rate_per_kwh"] == 0.28

    # Off-Peak: 02:00
    off_peak = get_tou_tariff(2)
    assert off_peak["tier"] == "Off-Peak"
    assert off_peak["rate_per_kwh"] == 0.18


def test_tou_economics_calculation():
    """Verify avoided coincident demand charge calculation."""
    # Uncontrolled surge = 150 kW, Scheduled cap = 100 kW -> 50 kW peak shaved
    econ = compute_tou_economics(uncontrolled_kw=150.0, scheduled_kw=100.0, hour=18)
    assert econ["tariff_tier"] == "On-Peak"
    # Avoided demand charge: 50 kW * $18.50/kW = $925.00
    assert econ["avoided_demand_charge"] == 925.0
    assert econ["monthly_projected_savings"] > 925.0


def test_temperature_safeguards(sample_ev_queue):
    """Verify cold plating derating (<10C) and extreme heat derating (>38C)."""
    # 1. Cold weather (<10C): Fast charging capped at <= 11 kW
    _, sched_cold = run_edf_scheduler(
        sample_ev_queue, cap_kw=120.0, ambient_temp_c=4.0
    )
    assert sched_cold["thermal_derated_count"] > 0
    assert sched_cold["ambient_temp_c"] == 4.0

    # 2. Extreme heat (>38C): Fast charging capped at <= 15 kW
    _, sched_hot = run_edf_scheduler(
        sample_ev_queue, cap_kw=120.0, ambient_temp_c=42.0
    )
    assert sched_hot["thermal_derated_count"] > 0
    assert sched_hot["ambient_temp_c"] == 42.0

    # 3. Normal ambient (22C): No thermal derate
    _, sched_norm = run_edf_scheduler(
        sample_ev_queue, cap_kw=120.0, ambient_temp_c=22.0
    )
    assert sched_norm["thermal_derated_count"] == 0


def test_ocpp_profile_export(sample_ev_queue):
    """Verify OCPP 1.6J / 2.0.1 compliant SetChargingProfile payload structure."""
    sched_df, _ = run_edf_scheduler(sample_ev_queue, cap_kw=80.0, ambient_temp_c=20.0)
    ocpp_payload = generate_ocpp_charging_profiles(
        sched_df, station_id="caltech_ST_001", timestamp_iso="2026-10-09T17:00:00Z"
    )

    assert ocpp_payload["station_id"] == "caltech_ST_001"
    assert ocpp_payload["ocpp_version"] == "1.6J / 2.0.1"
    assert len(ocpp_payload["set_charging_profile_requests"]) == len(sched_df)

    first_req = ocpp_payload["set_charging_profile_requests"][0]
    assert "connectorId" in first_req
    assert "csChargingProfiles" in first_req
    cs_profile = first_req["csChargingProfiles"]
    assert cs_profile["chargingProfilePurpose"] == "TxDefaultProfile"
    assert cs_profile["chargingSchedule"]["chargingRateUnit"] == "W"
    # Limits in watts should be >= 0
    assert cs_profile["chargingSchedule"]["chargingSchedulePeriod"][0]["limit"] >= 0.0

    # Verify JSON serializability
    serialized = json.dumps(ocpp_payload)
    assert len(serialized) > 100


def test_caltech_acn_data_loading_and_cleaning():
    """Verify real Caltech/JPL parquet can be ingested, cleaned, and feature engineered."""
    parquet_path = Path("EV-Charging-Demand-Forecasting-Capacity-Planning-Platform/data/processed/hourly_demand.parquet")
    if not parquet_path.exists():
        pytest.skip("ACN parquet not found")

    raw_df = pd.read_parquet(parquet_path)
    clean_df, summary = clean_pipeline(raw_df)

    assert "caltech_ST_001" in summary["stations"]
    assert len(clean_df) > 0

    feat_df = create_feature_dataset(clean_df)
    assert "temperature_c" in feat_df.columns
    assert "target_next_kw" in feat_df.columns
    assert len(feat_df) > 0
