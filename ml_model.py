"""
ml_model.py
-----------
Daily price forecast for the day after tomorrow (D+2) and the day after that (D+3).

Trains the final model (LightGBM, see ML_CONCEPT.md and 02_ml_model.ipynb) on the
latest 365 days and appends its predictions to the table "predictions" in
electricity_prices.db. The app reads them from there.

The pipeline is the same as in the notebook: local-time grid, fixed weather and
calendar features, asinh/MAD price scaling, prediction of the change vs. the last
known day. Only the evaluation is left out.

Run after BuildDB_v2.py (after approx. 13:00, when tomorrow's prices are known):
    python ml_model.py
Before 13:00 the last known day is today, so the script predicts tomorrow and the
day after instead.

Overview of the steps in this file:
  1. load_data / to_grid        read the database, reshape to "one row per day, 24 hour columns"
  2. wind_cf / weather_proxies  turn raw weather into simple production/demand indicators
     build_rows                 one row per (issue date, horizon, hour) with all fixed features
  3. AsinhMadScaler             scale prices so that spikes and negative prices don't dominate
     add_price_features         anchor (known day), target (change vs. known day), price features
  4. tree_inputs / fit_lightgbm prepare the inputs and train one LightGBM per horizon
  5. forecast / store           predict D+2 and D+3 and append them to the database

Terms used throughout:
  issue date D   the day on which the forecast is made (at approx. 13:00)
  anchor D+1     tomorrow: the last day whose prices are already known
  target D+h     the day to predict (h = 2 or 3)

AI notice: Created with Claude (Anthropic), reviewed and adapted by us.
"""

import sqlite3

import lightgbm as lgb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DATABASE = "electricity_prices.db"
TIME_FORMAT = "%Y-%m-%d %H:%M"          # same format as in BuildDB_v2.py
HORIZONS = (2, 3)                       # days after the issue date
WINDOW_DAYS = 365                       # training window, as in the walk-forward

# Weather columns in ml_data (each with suffix _fc_d1 / _fc_d2 / _fc_d3)
WEATHER = ["wind_de_hamburg", "wind_de_husum", "wind_de_emden", "solar_de", "solar",
           "temperature", "temperature_de"]
HOURS = list(range(24))                 # hours of a day in local time, 0-23
DAY_TYPES = ["workday", "saturday", "sunday_holiday"]   # the three kinds of days the model knows

# LightGBM settings: fixed ones and the result of the tuning in the notebook
#   learning_rate      small steps -> many trees, but more stable results
#   min_child_samples  a leaf needs at least 50 rows (protects against overfitting)
#   colsample_bytree / subsample: each tree sees only 80 % of the features / rows (more robust)
#   n_estimators       upper limit only; the real number of trees comes from early stopping
#   random_state       fixed seed, so the same data always gives the same model
LGBM_FIXED = dict(learning_rate=0.03, min_child_samples=50, colsample_bytree=0.8, subsample=0.8,
                  subsample_freq=1, n_estimators=3000, random_state=0, verbose=-1)
# Tuned per horizon in 02_ml_model.ipynb (tuning cell) and then frozen here.
# num_leaves / max_depth = how complex one tree may get, reg_lambda = L2 penalty
LGBM_TUNED = {2: dict(num_leaves=31, max_depth=6, reg_lambda=1),
              3: dict(num_leaves=31, max_depth=6, reg_lambda=10)}


# ---------------------------------------------------------------------------
# 1. Data on a local-time grid (one row per day, one column per hour 0-23)
# ---------------------------------------------------------------------------
def load_data():
    """Reads ml_data and the German prices and adds local date and hour."""
    with sqlite3.connect(DATABASE) as connection:
        data = pd.read_sql("SELECT * FROM ml_data", connection)
        prices_de = pd.read_sql("SELECT time_utc, price_rp_kwh FROM prices_de", connection)
    # German prices only have UTC times: derive Swiss local time the same way as BuildDB_v2
    # (convert UTC -> Zurich time, then drop the time-zone info so both tables match)
    prices_de["time_local"] = (pd.to_datetime(prices_de["time_utc"], utc=True)
                               .dt.tz_convert("Europe/Zurich").dt.tz_localize(None))
    # Add the local calendar day ("date") and the local hour (0-23) to both tables.
    # These two columns are what the grid below is built on.
    for frame in (data, prices_de):
        frame["time_local"] = pd.to_datetime(frame["time_local"])
        frame["date"] = frame["time_local"].dt.normalize()   # normalize = set time to 00:00
        frame["hour"] = frame["time_local"].dt.hour
    return data, prices_de


def to_grid(frame, column):
    """One row per local date, one column per hour. The 25-hour day averages the two
    02:00 hours, the 23-hour day interpolates the missing 02:00."""
    # pivot_table: long table (one row per hour) -> wide table (one row per day).
    # aggfunc="mean" handles the October clock change, where 02:00 occurs twice.
    grid = frame.pivot_table(index="date", columns="hour", values=column, aggfunc="mean")
    # In March the clock jumps from 02:00 to 03:00, so hour 2 is missing. Fill exactly
    # one missing hour between two known ones (limit=1, only "inside" the day).
    return grid.interpolate(axis=1, limit=1, limit_area="inside")


# ---------------------------------------------------------------------------
# 2. Fixed features (nothing fitted)
# These features only depend on weather and calendar, not on the training data,
# so they can be computed once for all days without any risk of data leakage.
# ---------------------------------------------------------------------------
def wind_cf(speed):
    """Wind speed (m/s) -> rough share of wind-turbine capacity (0-1)."""
    # Wind power grows with the cube of the wind speed, up to the rated speed
    # (~12 m/s), where the turbine produces its full capacity.
    share = np.minimum(speed, 12) ** 3 / 12 ** 3
    return share.mask((speed < 3) | (speed > 25), 0)   # below cut-in or above cut-out


def weather_proxies(data, lead):
    """Generation and demand proxies from the weather forecast made `lead` days ahead."""
    # One grid (day x hour) per weather column, using the forecast with the given lead time
    w = {column: to_grid(data, f"{column}_fc_d{lead}") for column in WEATHER}
    return {
        # Wind in northern Germany -> share of wind-turbine capacity, per location
        "wind_cf_hamburg": wind_cf(w["wind_de_hamburg"]),
        "wind_cf_husum": wind_cf(w["wind_de_husum"]),
        "wind_cf_emden": wind_cf(w["wind_de_emden"]),
        # Radiation (W/m²) / 1000 -> rough share of solar capacity (1000 W/m² = full sun)
        "solar_cf_de": (w["solar_de"] / 1000).clip(upper=1),
        "solar_cf_ch": (w["solar"] / 1000).clip(upper=1),
        # Heating degrees: how far the average CH/DE temperature is below 15 °C.
        # 0 when it is warm; the colder, the more heating demand.
        "hdd": (15 - (w["temperature"] + w["temperature_de"]) / 2).clip(lower=0),
    }


# Feature lists. The model sees exactly these columns, in this order.
WEATHER_FEATURES = ["wind_cf_hamburg", "wind_cf_husum", "wind_cf_emden",
                    "solar_cf_de", "solar_cf_ch", "hdd"]
PRICE_FEATURES = ["shape_lag", "shape_lag_week", "de_spread", "anchor_prev_diff"]
CATEGORICAL = ["hour", "day_type", "anchor_day_type"]
# For every weather feature there are three versions:
#   name        value in this hour of the target day
#   name_day    average of the whole target day
#   d_name      change of the daily average vs. the known day D+1
FEATURES = (CATEGORICAL + WEATHER_FEATURES + [f"{n}_day" for n in WEATHER_FEATURES]
            + [f"d_{n}" for n in WEATHER_FEATURES] + ["month_sin", "month_cos"] + PRICE_FEATURES)

# Monotone constraints: more wind or sun can only lower the price, heating demand only raise it
# One entry per feature in FEATURES: -1 = may only decrease the prediction,
# +1 = may only increase it, 0 = no constraint.
DOWN = [c for c in WEATHER_FEATURES if c.startswith(("wind", "solar"))]
MONOTONE = [-1 if f in DOWN or f in [f"d_{c}" for c in DOWN] else
            1 if f in ("hdd", "d_hdd") else 0 for f in FEATURES]


def build_rows(data, issues):
    """One row per (issue date D, horizon h, hour) with the fixed features."""
    # Day type of every date: Sunday or public holiday / Saturday / normal workday
    holiday = data.groupby("date")["holiday_ch"].max()
    weekday = holiday.index.dayofweek                  # 0 = Monday ... 6 = Sunday
    day_type = pd.Series(np.select([(weekday == 6) | (holiday.to_numpy() == 1), weekday == 5],
                                   ["sunday_holiday", "saturday"], "workday"), index=holiday.index)
    anchor_grids = weather_proxies(data, 1)            # weather of the known day D+1

    parts = []
    for h in HORIZONS:
        target_grids = weather_proxies(data, h)         # weather of the target day D+h
        # Every combination of issue date and hour (cross join) -> 24 rows per issue date
        rows = pd.DataFrame({"issue": issues}).merge(pd.DataFrame({"hour": HOURS}), how="cross")
        rows["h"] = h
        rows["anchor"] = rows["issue"] + pd.Timedelta(days=1)   # D+1, the last known day
        rows["target"] = rows["issue"] + pd.Timedelta(days=h)   # D+h, the day to predict
        # (target day, hour) pairs, used to look up one value per row in the grids
        cell = pd.MultiIndex.from_arrays([rows["target"], rows["hour"]])
        for name in WEATHER_FEATURES:
            # stack() turns the day x hour grid back into a long series with (day, hour)
            # index; reindex(cell) picks the value of each row's target day and hour
            rows[name] = target_grids[name].stack().reindex(cell).to_numpy()
            target_mean = target_grids[name].mean(axis=1).reindex(rows["target"]).to_numpy()
            anchor_mean = anchor_grids[name].mean(axis=1).reindex(rows["anchor"]).to_numpy()
            rows[f"{name}_day"] = target_mean                   # level of the target day
            rows[f"d_{name}"] = target_mean - anchor_mean       # change vs. the known day
        rows["day_type"] = day_type.reindex(rows["target"]).to_numpy()
        rows["anchor_day_type"] = day_type.reindex(rows["anchor"]).to_numpy()
        # Month as a point on a circle (sin/cos), so that December and January are
        # close to each other instead of 11 apart
        month = rows["target"].dt.month
        rows["month_sin"] = np.sin(2 * np.pi * month / 12)
        rows["month_cos"] = np.cos(2 * np.pi * month / 12)
        parts.append(rows)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Price scaling, anchor, target and price features
# ---------------------------------------------------------------------------
class AsinhMadScaler:
    """Robust price scaling: squashes spikes and negative prices without removing them."""

    def fit(self, prices):
        # Learn the centre (median) and spread (MAD) of the prices. Median and MAD
        # are hardly affected by a few extreme hours, unlike mean and std.
        values = prices.stack().dropna()
        self.m = values.median()
        self.s = 1.4826 * (values - self.m).abs().median()   # 1.4826: makes MAD comparable to std
        return self

    def transform(self, prices):
        # asinh behaves like a normal line near 0 and like a log for large values,
        # and (unlike log) it also works for negative prices
        return np.arcsinh((prices - self.m) / self.s)

    def inverse(self, z):
        # Back from the scaled value z to Rp./kWh
        return self.m + self.s * np.sinh(z)


def add_price_features(rows, scaler, price_ch, price_de):
    """Anchor A_D (mean scaled price of D+1), target y = z(D+h) - A_D and price features."""
    z_ch, z_de = scaler.transform(price_ch), scaler.transform(price_de)
    z_long, day_mean = z_ch.stack(), z_ch.mean(axis=1)

    def at(day):
        """Scaled CH price at the given day and each row's hour."""
        return z_long.reindex(pd.MultiIndex.from_arrays([day, rows["hour"]])).to_numpy()

    out = rows.copy()
    # Anchor: average scaled price of the known day D+1. All prices below are
    # expressed relative to it, so the model learns changes, not absolute levels.
    out["A"] = day_mean.reindex(out["anchor"]).to_numpy()
    # Target: how much the target hour differs from the known day's average
    out["y"] = at(out["target"]) - out["A"]                    # empty for the days to predict
    # Same hour on the known day D+1 (daily shape of yesterday's prices)
    out["shape_lag"] = at(out["anchor"]) - out["A"]
    # Same hour one week before the target day (weekly rhythm)
    out["shape_lag_week"] = at(out["target"] - pd.Timedelta(days=7)) - out["A"]
    # How much more expensive Germany was than Switzerland on the known day
    out["de_spread"] = z_de.mean(axis=1).reindex(out["anchor"]).to_numpy() - out["A"]
    # Trend: did the price level rise or fall from day D to day D+1?
    out["anchor_prev_diff"] = out["A"] - day_mean.reindex(out["issue"]).to_numpy()
    return out


# ---------------------------------------------------------------------------
# 4. LightGBM
# ---------------------------------------------------------------------------
def tree_inputs(rows):
    """The raw features, calendar columns as categories."""
    X = rows[FEATURES].copy()
    # Categorical with a fixed list of categories: LightGBM then treats "hour" as
    # 24 separate groups (not as a number), and training and prediction use the
    # same coding even if a category is missing in one of them.
    X["hour"] = pd.Categorical(X["hour"], categories=HOURS)
    for column in ("day_type", "anchor_day_type"):
        X[column] = pd.Categorical(X[column], categories=DAY_TYPES)
    return X


def fit_lightgbm(train, h):
    """Number of trees by early stopping on the last 60 days, then refit on all rows."""
    X, y = tree_inputs(train), train["y"]
    # True for the rows of the last 60 issue dates (used as validation set)
    last_60 = (train["issue"] > train["issue"].max() - pd.Timedelta(days=60)).to_numpy()
    settings = {**LGBM_FIXED, **LGBM_TUNED[h], "monotone_constraints": MONOTONE}
    # Probe model: train on the older rows, stop adding trees once the error on the
    # last 60 days has not improved for 100 trees. "l1" = mean absolute error.
    probe = lgb.LGBMRegressor(**settings).fit(
        X[~last_60], y[~last_60], eval_set=[(X[last_60], y[last_60])], eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False)])
    # Final model: same settings, the number of trees found above, trained on ALL
    # rows (so that the most recent 60 days are also learned from)
    return lgb.LGBMRegressor(**{**settings, "n_estimators": probe.best_iteration_}).fit(X, y)


# ---------------------------------------------------------------------------
# 5. Forecast
# ---------------------------------------------------------------------------
def forecast(issue=None):
    """
    Trains on the 365 days before the issue date and predicts D+2 and D+3.
    issue: None = the latest possible issue date (the day before the last day with
           all 24 prices). A past date can be given to check the model.
    Returns: one row per UTC hour with time_utc, target date, hour, predicted price.
    """
    data, prices_de = load_data()
    price_ch, price_de = to_grid(data, "price_rp_kwh"), to_grid(prices_de, "price_rp_kwh")

    # Find the issue date: the last day with all 24 prices is the anchor D+1,
    # so the issue date D is one day earlier
    complete = price_ch.index[price_ch.notna().all(axis=1)]    # days with all 24 prices
    issue = complete.max() - pd.Timedelta(days=1) if issue is None else pd.Timestamp(issue)
    anchor = issue + pd.Timedelta(days=1)
    window_start = issue - pd.Timedelta(days=WINDOW_DAYS)

    # Scaler fitted on the training window only, like in every walk-forward step
    scaler = AsinhMadScaler().fit(price_ch[(price_ch.index >= window_start) & (price_ch.index < issue)])
    # All issue dates from the start of the window up to today's issue date
    issues = pd.date_range(window_start, issue, freq="D")
    rows = add_price_features(build_rows(data, issues), scaler, price_ch, price_de)

    # Training rows: only targets known at the issue date (same purge rule as the walk-forward)
    # (a target day after the anchor would be a price we do not know yet -> leakage)
    train = rows[(rows["issue"] < issue) & (rows["target"] <= anchor)].dropna(subset=FEATURES + ["y"])
    # Rows to predict: today's issue date, for both horizons and all 24 hours
    today = rows[rows["issue"] == issue]
    # Stop with a clear message if an input is missing (e.g. weather forecast not loaded)
    missing = today[today[FEATURES].isna().any(axis=1)]
    if not missing.empty:
        raise ValueError(f"Features missing for {missing['target'].dt.date.unique()}: "
                         "run BuildDB_v2.py first (prices or weather forecast not available).")

    parts = []
    for h in HORIZONS:
        # One separate model per horizon (D+2 and D+3 behave differently)
        model = fit_lightgbm(train[train["h"] == h], h)
        rows_h = today[today["h"] == h]
        y_hat = model.predict(tree_inputs(rows_h))
        # Prediction is a change vs. the anchor in scaled units:
        # add the anchor back, then undo the scaling -> price in Rp./kWh
        parts.append(rows_h[["target", "hour"]].assign(
            predicted_rp_kwh=scaler.inverse(y_hat + rows_h["A"].to_numpy()).round(3)))
    predictions = pd.concat(parts)

    # Back to UTC hours. Merging on the real hours of ml_data handles the clock change:
    # the 25-hour day gets the 02:00 prediction twice, the 23-hour day has no 02:00.
    hours = data[["time_utc", "date", "hour"]].rename(columns={"date": "target"})
    return hours.merge(predictions, on=["target", "hour"]).sort_values("time_utc")


def store(predictions):
    """Appends the predictions to the table "predictions" (never overwrites old ones:
    the app compares them with the actual prices later)."""
    # All predictions of one run share the same creation time (part of the primary key)
    created_at = pd.Timestamp.now(tz="UTC").strftime(TIME_FORMAT)
    with sqlite3.connect(DATABASE) as connection:
        connection.executemany(
            "INSERT OR REPLACE INTO predictions (created_at, time_utc, predicted_rp_kwh) VALUES (?, ?, ?)",
            [(created_at, t, p) for t, p in zip(predictions["time_utc"], predictions["predicted_rp_kwh"])])
    return created_at


# Start by hand: python ml_model.py
if __name__ == "__main__":
    result = forecast()
    created_at = store(result)
    # Short summary per predicted day: average, cheapest and most expensive hour
    days = result.groupby(result["target"].dt.date)["predicted_rp_kwh"].agg(["mean", "min", "max"])
    print(f"Stored {len(result)} predicted hours (created_at {created_at} UTC):")
    print(days.round(2).to_string())
