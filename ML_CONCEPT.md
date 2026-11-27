# ML Concept: Price Model and Charging Plan


## Overview

Every day at 13:00 (Swiss time) the model predicts all 24 hourly Swiss day-ahead
prices of the **day after tomorrow** and the **day after that**, *relative to the
last known day*. A small linear program per vehicle then decides how many kWh to
charge in which hour of the vehicle's downtime. Models are validated walk-forward,
and the final verdict is a policy backtest: how much money the plan would have saved.

| Term | Definition |
|---|---|
| Decision time | 13:00 local on issue date D, after the day-ahead prices for D+1 are published |
| Delivery day | 24 hourly slots, hour 0–23 in Swiss local time (the market's own unit) |
| Known at 13:00 | all hours of D and D+1 |
| Anchor day | D+1, the most recent fully known delivery day |
| Horizon h | h = 2 (D+2) and h = 3 (D+3), all 24 hours predicted |
| Prediction unit | one row = (issue date D, horizon h, hour) |
| Target | y = z(price) − A_D, with z the scaled price and A_D the anchor (Step 3) |
| Charging hours | hours in which a vehicle is in the depot (Step 6) |

**Leakage rule:** every input must be known at 13:00 on D. Prices only from delivery
days ≤ D+1; weather only as forecasts, never measured values.

## Step 1: Data and grid

**Source:** table `ml_data` (one row per UTC hour). Prices are in Rp./kWh.

**Grid:** pivot to local date × hour 0–23, for prices (CH, DE) and weather:

```python
df["time_local"] = pd.to_datetime(df["time_local"])
df["date"], df["hour"] = df["time_local"].dt.date, df["time_local"].dt.hour
grid = df.pivot_table(index="date", columns="hour", values="price_rp_kwh", aggfunc="mean")
grid = grid.interpolate(axis=1, limit=1)
```

- 25-hour day (October): the two 02:00 rows land in one cell and are averaged.
- 23-hour day (March): the empty 02:00 cell is interpolated from 01:00 and 03:00.
- All price features are built from this grid, **not** from the `price_lag_*`
  columns of `ml_data` (those are shifted in UTC and are off by one local hour
  around the clock change).

**Weather per row (D, h, hour):**

| Needed | Column in `ml_data` |
|---|---|
| target day D+2 | `*_fc_d2` |
| target day D+3 | `*_fc_d3` |
| anchor day D+1 (for the deltas) | `*_fc_d1` |

`_fc_dN` = forecast made N×24 h before the target hour. For evening hours this is
a few hours after 13:00 (known, small inaccuracy, see README). Target-day and
anchor-day weather come from *different* forecast runs, not the same one.

**Start:** first issue date with complete weather features is **9 March 2024**.

## Step 2: Fixed features (no fitting, cannot leak)

| Feature | Formula | Why |
|---|---|---|
| `hour` | categorical 0–23 | daily shape |
| `wind_cf_hamburg`, `wind_cf_husum`, `wind_cf_emden` | per location: 0 if v < 3 or v > 25 m/s, else min(v, 12)³ / 12³ | wind generation proxy (0–1); one column per location |
| `solar_cf_de`, `solar_cf_ch` | min(radiation / 1000, 1) for `solar_de` and `solar` (CH) | solar generation proxy; drives midday dip and negative prices |
| `hdd` | max(0, 15 − T), T = mean of `temperature` (CH) and `temperature_de` | heating load |
| `*_day` (each wind and solar column, `hdd`) | mean over the target day | level drivers |
| `d_*` (each wind and solar column, `hdd`) | target-day mean − anchor-day mean | change vs. the known day |
| `day_type` | workday / Saturday / Sunday-or-holiday (holidays: canton Zurich) | weekend and holiday effect |
| `anchor_day_type` | same, for D+1 | the anchor may itself be a cheap weekend day |
| `is_charge_hour` | 1 if any vehicle can charge in this hour (definition in Step 5) | the decisive metrics |
| `month_sin`, `month_cos` | sin / cos of 2π·month / 12 | season without a Dec→Jan jump |

Wind and solar locations stay **separate columns**: the models learn how much
weight each region gets, instead of a hand-picked weighting. Wind cut-in 3,
rated 12, cut-out 25 m/s are assumptions.

## Step 3: Price scaling and target (refitted in every walk-forward step)

1. **Scaler**, fitted on CH prices (all hours) of the training window only:
   m = median(p), s = 1.4826 · median(|p − m|)
2. **Transform** every price (CH and DE, both with the CH m and s):
   z = asinh((p − m) / s)
3. **Anchor:** A_D = mean z over the 24 hours of D+1
4. **Target:** y(D, h, k) = z(D+h, hour k) − A_D
5. **Price features:**
   - `shape_lag` = z(D+1, k) − A_D (the anchor day's shape)
   - `shape_lag_week` = z(D+h−7, k) − A_D (same weekday one week earlier)
   - `de_spread` = mean z(DE) of D+1 − A_D
   - `anchor_prev_diff` = A_D − A_(D−1) (momentum of the level)

Never drop spikes or negative prices; asinh compresses them. Never fit m and s on
the full history.

## Step 4: Models

All fitted separately for h = 2 and h = 3. Each must beat the ones above it.
The models stay **separate**: no averaging (ensemble) of Elastic Net and LightGBM.

| # | Model | Prediction / setup |
|---|---|---|
| 0a | Copy of D+1 | y = `shape_lag` (price of D+h = price of D+1, hour by hour) |
| 0b | Last week | y = `shape_lag_week` |
| 1 | Benchmark profile | mean y of training rows with the same (hour, `day_type`, `anchor_day_type`); fewer than 8 rows → fall back to hour only |
| 2 | Elastic Net | one-hot `hour` + weather, calendar and price features + interactions hour × each wind column, hour × each solar column, hour × `hdd`, hour × `day_type`; `StandardScaler` in an sklearn `Pipeline`; `ElasticNetCV`, l1_ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 1.0}, 100 alphas, max_iter 10 000; inner CV split by issue date with a 4-day gap (never plain `KFold`) |
| 3 | LightGBM | same features, `hour` categorical, no hand-made interactions; fixed: learning_rate 0.03, min_data_in_leaf 50, feature_fraction 0.8, bagging_fraction 0.8; grid (27 combinations): num_leaves ∈ {7, 15, 31}, max_depth ∈ {3, 4, 6}, lambda_l2 ∈ {1, 5, 10}, tuned once on the first year of the development period, then frozen (result: 31 / 6 / 1 for D+2, 31 / 6 / 10 for D+3); number of trees by early stopping on the last 60 days of each training window; monotone constraints −1 on all wind and solar columns and their `d_*`, +1 on `hdd`, `d_hdd` |

Simple-rule errors (MAE in Rp./kWh, local-time grid, charging hours 16:00–08:00):
copy of D+1 1.24 (h = 2) / 1.61 (h = 3); last week 1.62 (both). The models must
beat the better rule per horizon.

Known limitation: LightGBM cannot predict beyond the range of y seen in training.
This matters less because y is the change vs. the known day, not the price itself.

## Step 5: Walk-forward evaluation

| Period | Dates | Use |
|---|---|---|
| Development | 10 Mar 2025 to 19 May 2026 | all model and setting choices |
| Hold-out | 20 May 2026 to latest | touched **once**, at the very end |

Loop:
1. Step start D₀, step length **1 month** (monthly refit).
2. Training window: issue dates in [D₀ − 365 days, D₀).
3. Purge: keep a training row (d, h) only if d + h ≤ D₀ + 1.
4. Fit the scaler (Step 3), transform, fit the models on the training rows.
5. Predict all issue dates of the month for h = 2, 3, all 24 hours.
6. Move D₀ one month forward.

**Charging hours** for the metrics: every hour in which *any* vehicle of the fleet
can charge, read from the `vehicles` and `schedule` tables: from the earliest
`depot_arrival` to the latest `depot_departure` (sample fleet: 16:00–08:00), and
the whole day on days without driving (sample fleet: Sunday). The model serves
the whole fleet; Van 1 alone is only used in the policy backtest (Step 7).

**Metrics** per model and h, on all hours **and** on charging hours (the
charging-hour score decides):

| Metric | Measures |
|---|---|
| MAE of the cheapest-4-hours level | is a night cheaper or dearer than the known one? |
| Sign accuracy of the level change | "cheaper than the known night" right? |
| Spearman rank correlation | are the hours ranked correctly within a night? |
| Hourly MAE (Rp./kWh) | general accuracy |

**Back to prices:** p̂ = m + s · sinh(ŷ + A_D), with the same step's m, s and A_D.

## Step 6: Charging plan (LP per vehicle)

There are enough charging stations for all vehicles, so each vehicle gets its own
small LP. Data comes from the app tables `vehicles`, `vehicle_types`, `schedule`
and `tariffs`.

**Charging hours (u_t = 1):** from `depot_arrival` to `depot_departure`, and the
whole day on days with 0 km in `schedule` (sample fleet: Sunday, so Saturday
evening to Monday morning is one window). Not a fixed 21:00–06:00 window.

**Costs c_t per hour:**
- days D and D+1: actual day-ahead prices
- days D+2, D+3: predicted prices
- plus the tariff surcharge (`surcharge_rp_kwh`, peak surcharge between
  `peak_start` and `peak_end`), because charging can now fall into daytime hours

**LP** over the charging hours of windows j = 0 (tonight), 1, 2; x_t = kWh from the grid:

    minimise   Σ c_t · x_t
    subject to E_dep(j)   = E_arr(j) + η · Σ_{t in j} x_t
               E_arr(j+1) = E_dep(j) − C(j+1)
               C(j+1) + E_reserve ≤ E_dep(j) ≤ E_max
               0 ≤ x_t ≤ P_max · 1 h · u_t
               E_dep(2) ≥ E_terminal

| Symbol | From |
|---|---|
| E_arr(0) | `battery_percent` × `battery_kwh` |
| C(j) | km of that weekday (`schedule`) × `consumption_kwh_100km` / 100 |
| E_max | `battery_kwh` |
| E_reserve | `min_reserve_percent` × `battery_kwh` |
| P_max | `charging_power_kw` |
| η | 0.9 (charging efficiency, assumption) |

The last window is cut at the end of D+3 (no prices beyond). The terminal
constraint stops the LP from pushing all charging past the horizon. Solve with
`scipy.optimize.linprog(method="highs")`. Only tonight's plan is executed;
the next day at 13:00 everything is solved again with fresh prices.

## Step 7: Policy backtest (final verdict)

Replay every day of the test period for one sample vehicle (Van 1): at 13:00
build the plan, execute tonight at **actual** prices, update the battery with that
day's driving, repeat.

| Policy | What it does |
|---|---|
| A · Fixed window | charge every night 01:00–05:00, just enough for the next day |
| B · Threshold | charge tonight if its cheapest 4 h are below the rolling 30-day median, else only the minimum |
| C · Benchmark + LP | benchmark profile forecasts → LP |
| D · Elastic Net + LP | Elastic Net forecasts → LP |
| E · LightGBM + LP | LightGBM forecasts → LP |
| F · Perfect foresight | LP with actual prices: the upper bound |

**Key number:** share of achievable savings captured = (cost_A − cost_X) / (cost_A − cost_F).
Also: average cost per kWh charged, number of reserve violations (must be 0).

## Step 8: Daily run and app

1. `BuildDB_v2.py` updates the database (after 13:00).
2. The ML script builds the rows for today's issue date, predicts D+2 and D+3
   with the chosen model and converts them back to Rp./kWh.
3. It **appends** them to the table `predictions` (`created_at`, `time_utc`,
   `predicted_rp_kwh`); never overwrite, the app's "Prediction quality" tab needs
   the old ones.
4. The app solves the LP per vehicle and shows the plan.

Models stay fixed in live use; retraining is a manual step.

## Files

| File | Content |
|---|---|
| `01_data_analysis.ipynb` | preliminary data analysis: data quality, drivers, where the savings are |
| `02_ml_model.ipynb` | first version, run step by step: grid, features, models, walk-forward (Steps 1–5) |
| `ml_model.py` | later: the final code from the notebook as a function for the daily run |

## Not in version 1

| Item | Reason |
|---|---|
| EEX futures | no data access |
| Ensemble (mean of Elastic Net and LightGBM) | models are compared separately |
| Quantile models, planning with q70 | later, if time allows |
| 730-day training window | only testable on a short period |
| Night weighting (`sample_weight`) | a setting to test later |
| Weekly refit | monthly is 4× faster with the same logic |
| German holidays (`holiday_de`) as a feature, interaction solar × Sunday/holiday | potential improvement: sunny spring Sundays and holidays with negative prices are predicted 3–6 Rp./kWh too expensive (worst days of the backtest, e.g. Easter 2026) |

## Open points

- Who writes the LP: ML (this concept) or the app team in `ladeplanung.py`? Agree on one.
- Policy backtest for one van (planned) or the whole fleet.
