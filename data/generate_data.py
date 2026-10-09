"""Synthetic EV charging dataset generator.

Generates realistic hourly EV demand time-series data for 3 stations
with morning/evening peaks, weekend drops, temperature correlations,
and deliberately injected missing values and outlier spikes.
"""

from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

# Constants
RANDOM_SEED = 42
NUM_DAYS = 40
STATION_IDS = ["Station_A", "Station_B", "Station_C"]
START_DATE = datetime(2025, 1, 1, 0, 0)
MISSING_PROBABILITY = 0.03   # ~3% missing values
OUTLIER_PROBABILITY = 0.01  # ~1% extreme outlier spikes
CHARGER_POWERS_KW = [7.4, 11.0, 22.0]


def generate_station_base_pattern(
    hours: np.ndarray,
    day_of_week: np.ndarray,
    station_multiplier: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes hourly demand and active sessions based on temporal curves."""
    # Morning commute peak (8-10 AM)
    morning_peak = 35.0 * np.exp(-0.5 * ((hours - 9) / 1.5) ** 2)
    # Evening return peak (6-9 PM)
    evening_peak = 55.0 * np.exp(-0.5 * ((hours - 19) / 2.0) ** 2)
    # Night base consumption (1-5 AM)
    night_base = 5.0 + 3.0 * np.sin(hours / 24.0 * 2 * np.pi)

    # Weekend reduction factor: ~35% drop in commercial charging
    is_weekend = (day_of_week >= 5).astype(float)
    weekend_factor = 1.0 - (0.35 * is_weekend)

    noise = rng.normal(0, 4.0, size=len(hours))
    demand = (night_base + morning_peak + evening_peak) * station_multiplier * weekend_factor + noise
    demand = np.clip(demand, 0.0, None)  # Demand cannot be negative

    # Active charging sessions strongly correlate with demand
    sessions = np.round(demand / rng.uniform(7.0, 11.0, size=len(hours))).astype(int)
    sessions = np.clip(sessions, 0, 30)

    return demand, sessions


def inject_anomalies(
    demand: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Injects ~3% missing values (NaNs) and ~1% extreme outlier spikes."""
    corrupted = demand.copy()
    n = len(corrupted)

    # Inject missing values (sensor disconnection / packet drops)
    missing_mask = rng.random(size=n) < MISSING_PROBABILITY
    corrupted[missing_mask] = np.nan

    # Inject extreme outlier spikes (meter glitch / surge error: 4x-6x demand)
    outlier_mask = rng.random(size=n) < OUTLIER_PROBABILITY
    # Only corrupt non-missing points
    valid_outliers = outlier_mask & (~np.isnan(corrupted))
    corrupted[valid_outliers] = corrupted[valid_outliers] * rng.uniform(4.0, 6.0, size=valid_outliers.sum()) + 150.0

    return corrupted


def generate_synthetic_data(
    num_days: int = NUM_DAYS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generates the primary multi-station hourly demand dataset."""
    rng = np.random.default_rng(seed)
    total_hours = num_days * 24
    timestamps = [START_DATE + timedelta(hours=i) for i in range(total_hours)]

    hours = np.array([ts.hour for ts in timestamps])
    days_of_week = np.array([ts.weekday() for ts in timestamps])

    # Seasonal/diurnal temperature curve (colder at night, warmer afternoon)
    base_temp = 14.0 + 7.0 * np.sin((hours - 8) / 24.0 * 2 * np.pi)
    temp_noise = rng.normal(0, 1.5, size=total_hours)
    temperatures = np.round(base_temp + temp_noise, 1)

    # Simplified holiday indicator (e.g. day 15 and 26)
    is_holiday = np.array([1 if (ts.day in [15, 26]) else 0 for ts in timestamps])

    station_multipliers = {"Station_A": 1.0, "Station_B": 1.35, "Station_C": 0.75}
    records = []

    for station_id in STATION_IDS:
        mult = station_multipliers[station_id]
        demand_raw, sessions = generate_station_base_pattern(hours, days_of_week, mult, rng)
        demand_anomalous = inject_anomalies(demand_raw, rng)

        for i in range(total_hours):
            records.append({
                "timestamp": timestamps[i].strftime("%Y-%m-%d %H:%M:%S"),
                "station_id": station_id,
                "demand_kw": np.round(demand_anomalous[i], 2) if not np.isnan(demand_anomalous[i]) else np.nan,
                "active_sessions": int(sessions[i]),
                "temperature_c": float(temperatures[i]),
                "is_holiday": int(is_holiday[i]),
            })

    df = pd.DataFrame(records)
    # Sort chronologically by station
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["station_id", "timestamp"]).reset_index(drop=True)
    return df


def generate_ev_queue(
    slot_timestamp: datetime,
    station_id: str,
    n_evs: int = 8,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generates a realistic queue of waiting EVs for a specific time slot.

    Includes EV ID, arrival time, deadline hour, requested kWh, and onboard max charger kW.
    """
    rng = np.random.default_rng(seed if seed is not None else int(slot_timestamp.timestamp()) % 100000)
    queue = []

    for idx in range(1, n_evs + 1):
        ev_id = f"EV-{station_id[-1]}-{idx:02d}"
        # Deadline between 1 to 4 hours from slot
        deadline_lead = rng.integers(1, 5)
        deadline = (slot_timestamp + timedelta(hours=int(deadline_lead))).strftime("%Y-%m-%d %H:00")
        energy_needed = round(float(rng.uniform(8.0, 45.0)), 1)
        max_kw = float(rng.choice(CHARGER_POWERS_KW))

        battery_capacity = float(rng.choice([50.0, 64.0, 77.0, 82.0, 100.0]))
        soc_pct = round(float(rng.uniform(22.0, 93.0)), 1)

        queue.append({
            "ev_id": ev_id,
            "arrival_time": slot_timestamp.strftime("%Y-%m-%d %H:00"),
            "deadline": deadline,
            "hours_remaining": int(deadline_lead),
            "energy_needed_kwh": energy_needed,
            "battery_capacity_kwh": battery_capacity,
            "soc_pct": soc_pct,
            "max_kw": max_kw,
        })

    return pd.DataFrame(queue)


def main() -> None:
    """CLI execution to generate and save standard ev_demand.csv."""
    data_dir = Path(__file__).resolve().parent
    out_path = data_dir / "ev_demand.csv"
    print(f"Generating synthetic EV demand dataset with seed {RANDOM_SEED}...")
    df = generate_synthetic_data(num_days=NUM_DAYS, seed=RANDOM_SEED)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path.resolve()}.")


if __name__ == "__main__":
    main()
