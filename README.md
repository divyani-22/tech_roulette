# ⚡ VoltCap: EV Charging Demand Forecast + Cap-Safe Scheduler

> **Predict the surge. Respect the cap.**  
> An end-to-end, production-grade ML forecasting and capacity-safe optimization platform designed for commercial EV charging hubs and local distribution grids.

---

## 🎯 1. Problem Statement
Commercial EV charging hubs face strict physical constraints dictated by local substation transformers and grid connection agreements. If simultaneous EV charging draws exceed this **Power Cap (kW)**, circuit breakers trip, local transformers overheat, and operators incur catastrophic peak-demand penalties.

**VoltCap** solves this dual problem:
1. **AI/ML Forecast:** Predicts aggregate charging station demand for the **next 1-hour time slot ($t+1$)** with empirical uncertainty confidence intervals.
2. **Cap-Safe Heuristic Optimization:** Schedules individual incoming EVs using a greedy **Earliest-Deadline-First (EDF)** algorithm under a rigid transformer capacity ceiling ($P_{\text{cap}}$), providing a **mathematical guarantee** that total power allocation never exceeds the cap.

---

## 🚀 2. Quickstart & How to Run

### Installation & Execution
```bash
# 1. Navigate to repository root
cd voltcap

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Generate the 40-day synthetic dataset with anomalies
python data/generate_data.py

# 4. Run the automated test suite
pytest tests

# 5. Launch the Streamlit Control Room application
streamlit run app.py
```

---

## 🖥️ 3. 7-Step Demo Flow

The application presents an intuitive 7-step numbered stepper designed for seamless hackathon presentations and vivas:

| Step | Tab | What is Displayed Live |
|---|---|---|
| **1** | **Data & Preprocessing** | Raw vs clean summary, missing imputation counts (≤2h linear interpolation), Tukey IQR outlier capping, and feature leakage rationales. |
| **2** | **Baseline** | Non-ML naive benchmarks: Baseline A (24-hour diurnal persistence) & Baseline B (3-hour rolling average). |
| **3** | **Training** | Live Random Forest training telemetry (runtime in seconds), chronological split defense, and Gini feature importances. |
| **4** | **Evaluation** | Comparative leaderboard: MAE & RMSE in kW (MAPE avoided), % improvement over baseline, and actual vs predicted line plots. |
| **5** | **Prediction** | Next-hour point forecast (kW) with 80% confidence interval derived from tree spread (10th–90th percentile) and grid headroom gauge. |
| **6** | **Product Decision** | Per-EV greedy EDF schedule with color-coded chips (`Charge Now`, `Throttle`, `Defer`), ToU electricity tariffs, multi-factor BMS safeguards (SoC ≥ 80% CV taper + cold/heat derating), avoided demand charge savings, and 1-click **OCPP 1.6J/2.0.1** profile export. |
| **7** | **Impact** | Uncontrolled surge vs scheduled peak shaving (kW and %), energy delivery %, avoided blackout metric, and plain-English takeaways. |

---

## 🧠 4. Machine Learning Architecture

### Target Formulation
- **Target ($y$):** Station demand in kW at time $t + 1$ (`target_next_kw`).

### Past-Only Feature Engineering (Zero-Leakage Guarantee)
Every feature uses information strictly available at or prior to observation time $t$:
- `hour`: Diurnal commute cycles (morning 8–10 AM & evening 6–9 PM peaks).
- `day_of_week`: Weekly commuting fluctuations (Monday through Sunday).
- `is_weekend`: Binary indicator for commercial charging drops.
- `lag_1`: Most recently recorded power demand at $t$ (immediate momentum).
- `lag_24`: Power demand recorded at $t-23$ relative to target $t+1$ (strong 24h seasonal periodicity).
- `rolling_mean_3`: 3-hour backward-looking rolling average ($t, t-1, t-2$) smoothing high-frequency noise.
- `active_sessions_lag_1`: Number of active physical plug sessions at $t$.
- `temperature_c`: Ambient temperature affecting EV battery chemistry and cabin preconditioning.

### Train/Test Splitting Strategy
- **Chronological 80/20 Split (Strictly Non-Shuffled):**
  > **Why not shuffle?** Random shuffling in time-series allows future timestamps to train past predictions. This violates causality and yields unrealistically optimistic performance metrics that immediately fail in real production deployments.

### Why Avoid MAPE?
- Night-time EV charging demand regularly approaches **0 kW**.
- $\text{MAPE} = \frac{1}{n}\sum \left|\frac{y - \hat{y}}{y}\right|$ produces numerical division by near-zero, inflating error metrics infinitely.
- VoltCap uses **MAE (kW)** and **RMSE (kW)**, reflecting genuine physical power deviations.

---

## ⚡ 5. Capacity-Safe Scheduler (Greedy EDF)

When aggregate requested EV power exceeds transformer cap $P_{\text{cap}}$:
1. Sort vehicles primarily by **earliest departure deadline** (`deadline` ascending).
2. Break ties by **energy required** (`energy_needed_kwh` descending).
3. For each vehicle in queue:
   $$P_{\text{alloc}} = \min(P_{\text{max\_charger}}, E_{\text{needed}}, P_{\text{remaining\_cap}})$$
   $$P_{\text{remaining\_cap}} \leftarrow P_{\text{remaining\_cap}} - P_{\text{alloc}}$$
4. Assign operational action:
   - $P_{\text{alloc}} = P_{\text{max\_charger}} \implies$ **Charge Now** (Green)
   - $0 < P_{\text{alloc}} < P_{\text{max\_charger}} \implies$ **Throttle** (Amber)
   - $P_{\text{alloc}} = 0 \implies$ **Defer** (Red)

### Mathematical Invariant
$$\sum_{i=1}^{N} P_{\text{alloc}, i} \le P_{\text{cap}}$$
Enforced via runtime assertions ensuring grid breaker trips are physically impossible.

---

## 🧪 6. Failure Cases & Resilience

VoltCap implements tests and interactive control panel buttons for three mission-critical failure modes:
1. **Extreme Power Surge (3× Cap):** 15+ concurrent high-power EVs arrive simultaneously. Scheduler caps draw at 100% capacity and defers flexible vehicles without failure.
2. **Missing Real-Time Sensor (`lag_1 = NaN`):** Telemetry outage. System catches the missing signal and gracefully falls back to diurnal persistence (`lag_24`), flagging an amber fallback pill.
3. **Malformed / Bad Schema Input:** Empty datasets, missing columns, or negative demand numbers are caught with graceful error cards.

---

## ⚠️ 7. Limitations & Future Scope
- **Current Scope:** Evaluates single-hub 1-hour discrete slot load balancing.
- **Future Scope:** Multi-phase unbalanced AC distribution modeling, vehicle-to-grid (V2G) reverse power injection, dynamic real-time electricity tariff co-optimization.

---

## 🎓 8. Viva One-Liners (Quick Examiner Answers)

1. **Why greedy EDF instead of full MILP or Reinforcement Learning?**  
   *“EDF runs in deterministic $\mathcal{O}(N \log N)$ time, provably minimizes maximum lateness for single-resource scheduling, and requires zero cloud solver licensing.”*
2. **Why is data never shuffled during train/test split?**  
   *“Random shuffling destroys temporal autocorrelation and causes lookahead data leakage, creating false high accuracy that breaks in production.”*
3. **How is prediction uncertainty calculated without Bayesian neural nets?**  
   *“We query individual predictions across all 200 trees in the Random Forest ensemble and calculate the empirical 10th and 90th percentiles live.”*
4. **Why MAE instead of MAPE?**  
   *“EV demand at 3 AM is frequently 0 kW; dividing by zero causes infinite MAPE distortion, whereas MAE measures physical kilowatts.”*
5. **How do you guarantee the cap is never exceeded?**  
   *“The remaining capacity monotonically decrements with each allocation, backed by a strict hard invariant assertion at the end of the optimization loop.”*
