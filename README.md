# Smart Charging for Electric Fleets

A Streamlit app that plans **when** a company's electric vehicles should charge so that charging is as cheap as possible, while every vehicle still has enough battery for its next trip.

## The problem

Companies with electric vans (e.g. parcel services) usually charge their vehicles overnight in the depot. Electricity prices on the Swiss day-ahead market change every hour and can differ by a factor of ten within a single day. Some nights are cheap, others are not.

The exchange only publishes prices **up to tomorrow** (around 13:00). To plan charging over a weekend or for vehicles that stand in the depot for several days, the app also needs prices for the **day after tomorrow** and the **day after that**. Our own machine-learning model predicts these two days.

**Terms used throughout the project:**

| Term | Meaning |
|---|---|
| Day 0 | today |
| Day 1 | tomorrow, prices known from approx. 13:00 |
| Day 2, Day 3 | predicted by the ML model |

---

## Installation and start

Requirements: **Python 3.10 or newer**.

```bash
pip install -r requirements.txt
python BuildDB_v2.py                   # builds electricity_prices.db (first run: a few minutes)
python ml_model.py                     # predicts D+2 and D+3, writes them to the table "predictions"
python -m streamlit run app.py         # starts the app
```

All files must be in the same folder as `electricity_prices.db`. Run `BuildDB_v2.py` and then `ml_model.py` daily after approx. 13:00. Before that, tomorrow's prices are not published yet.

---

## Part 1: Data preparation (`BuildDB_v2.py`)

`BuildDB_v2.py` downloads all market and weather data, prepares the features for the ML model and stores everything in one SQLite database, `electricity_prices.db`. It also creates the tables the app works with.

### Data sources

| Data | Source | License |
|---|---|---|
| Day-ahead prices Switzerland (`CH`) and Germany/Luxembourg (`DE-LU`) | [Energy-Charts API](https://api.energy-charts.info), Fraunhofer ISE | CC BY 4.0 |
| Measured weather (archive) | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | CC BY 4.0 |
| Current weather forecast | [Open-Meteo Forecast API](https://open-meteo.com/en/docs) | CC BY 4.0 |
| Weather as it was forecast in the past | [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) | CC BY 4.0 |
| Public holidays CH (canton of Zurich) and DE (nationwide) | Python package [`holidays`](https://pypi.org/project/holidays/) | MIT |

Prices are converted from €/MWh to **Rp./kWh**, the unit on every Swiss electricity bill, with a fixed exchange rate of 0.94 CHF/EUR. Since October 2025 the exchange publishes 15-minute prices; we average them to hourly values.

### Weather locations

| Group | Locations | Use |
|---|---|---|
| Switzerland | Zurich, Bern, Geneva, Basel, Lugano | temperature, wind at 100 m, solar radiation, **averaged** |
| Northern Germany | Hamburg, Husum (North Sea coast), Emden (East Frisia) | wind at 100 m only, **one column per location** |
| German solar regions | Bavaria, Baden-Württemberg, North Rhine-Westphalia, Saxony | solar radiation only, **averaged** (`solar_de`) |
| German cities | Frankfurt, Berlin, Munich | temperature only, **averaged** (`temperature_de`) |

Germany has by far the most solar capacity in the coupled market; sunny days there push midday prices down, often below zero. German heating demand moves prices as well. The German averages are only filled when all locations of the group are present.

The large German wind farms are in the north. Strong wind there lowers prices across central Europe, including Switzerland. The three locations are kept separate so that the model can decide which region matters most. Wind is taken at 100 m (hub height of wind turbines) and always in m/s.

### Tables in `electricity_prices.db`

**Market data:** updated on every run. Stored values are reused and only the last 14 days are reloaded.

| Table | Content |
|---|---|
| `prices_ch` | Swiss day-ahead price per hour |
| `prices_de` | German day-ahead price per hour |
| `weather` | measured weather (forecast for hours not yet in the archive) |
| `weather_forecasts` | weather as it was forecast 1, 2 and 3 days in advance |
| `ml_data` | everything the ML model needs, one row per hour |

**App tables:** filled with sample data on the first run only, never overwritten afterwards.

| Table | Content |
|---|---|
| `vehicles`, `vehicle_types`, `schedule` | fleet, vehicle models, km per weekday |
| `tariffs`, `settings` | surcharges of the local utility, user settings |
| `predictions` | ML predictions with creation time (written by the ML script) |
| `charging_plan` | saved charging plans |

All times are stored in **UTC** as text (`YYYY-MM-DD HH:MM`). Calendar features use Swiss local time.

### Columns in `ml_data`

| Column(s) | Meaning |
|---|---|
| `time_utc`, `time_local` | hour (not features) |
| `price_rp_kwh` | **label**: Swiss price of this hour, empty for hours still to be predicted |
| `hour`, `weekday`, `weekend`, `month` | calendar |
| `holiday_ch`, `holiday_de` | 1 = public holiday (CH: canton of Zurich; DE: nationwide) |
| `price_lag_24h`, `price_lag_48h`, `price_lag_168h` | Swiss price 1, 2 and 7 days earlier |
| `price_de_lag_24h`, `price_de_lag_48h`, `price_de_lag_168h` | German price 1, 2 and 7 days earlier |
| `temperature`, `wind_100m`, `solar`, `wind_de_hamburg`, `wind_de_husum`, `wind_de_emden`, `solar_de`, `temperature_de` | **measured** weather (for comparison only) |
| same columns with suffix `_fc_d1` / `_fc_d2` / `_fc_d3` | weather as it was **forecast** 1 / 2 / 3 days in advance. `_fc_d1` serves the known day (day 1), so the model can compare the target day with it. |
| `is_forecast` | 1 = weather of this hour is only a forecast (not a feature) |

### Avoiding data leakage

The model may only use information that is known at **13:00 on day 0**:

- **No same-hour prices.** The Swiss or German price of the target hour is never stored as a feature, only shifted prices.
- **Two models.** The model for day 2 uses `*_lag_24h` (= tomorrow's known price). The model for day 3 must use `*_lag_48h` instead, because for day 3 the 24 h lag would be day 2, which is still unknown.
- **Forecast instead of measured weather.** At decision time only a weather *forecast* exists for days 2 and 3. Training on measured weather would make the model look better in the test than it really is. The `_fc_d2` / `_fc_d3` columns therefore contain historical forecasts from the Previous Runs API. For future hours they are filled with the current forecast.

### Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Weather model | `ecmwf_ifs025` (fixed, not "best match") | A fixed model keeps the source consistent over time. |
| Forecast lead time | `_fc_d1` / `_fc_d2` / `_fc_d3` = forecast from 1 / 2 / 3 days before the target hour | "2 days before" means 48 h before the target hour. For day 2 at 18:00 this is day 0 at 18:00, a few hours **after** the decision at 13:00. For weather forecasts the difference is small. The strict variant (3 and 4 days) can be set in `FORECAST_DAYS_AHEAD`. |
| Start of data | 1 January 2024 | One full year for training plus test months, so the model sees every season. |
| Storage | SQLite, prices in Rp./kWh | Shared by app, ML script and BuildDB_v2 without extra tools. |

### Automatic update

The app calls `BuildDB_v2.update_if_needed()` when it opens. If prices are missing (before 13:00: until end of today; after 13:00: until end of tomorrow), the database is updated in the background. Without internet, or if prices are not published yet, the app keeps running with the last known data and tries again after 30 minutes at the earliest. Requests that hit the API limit (HTTP 429) are retried automatically.

### Known limitations

- **Forecast data starts late.** In the Previous Runs API, `ecmwf_ifs025` provides temperature from 5 Feb 2024, but wind at 100 m and solar radiation only from **9 Mar 2024** (checked on 24 Nov 2026). Before that, the `_fc` columns are empty and these rows are dropped from training.
- **Daylight saving time.** Days with a time change have 23 or 25 hours; we do not force a 24-hour grid. This affects 2 days per year.
- **Radiation timestamp.** Open-Meteo reports solar radiation as the average of the *preceding* hour, so there may be an offset of one hour.
- **Fixed exchange rate.** EUR→CHF is fixed at 0.94.
- **Very first hour.** The weather of 1 Jan 2024, 00:00 (local time) is missing because the archive starts later.

### Deliberately not implemented

| Item | Reason |
|---|---|
| EEX futures | Not available. The ML script compares against simple baselines instead. |
| More wind locations (6 instead of 3) | The three northern German wind locations cover the most important effect. |
| Parquet files, raw data copies | Would break app, ML script and checks; SQLite is enough. |
| Prices in €/MWh | App and model work with Rp./kWh. |
| Gap log, number of quarter-hours per hour | Little benefit given very few gaps. |
| Prices from 2023 | Weather forecasts only start in 2024 anyway. |
| `price_lag_72h` (use before 13:00) | The app is used after 13:00. |

---

## Part 2: Machine-learning model

The full plan (grid, features, models, walk-forward evaluation, charging LP, policy backtest) is in [`ML_CONCEPT.md`](ML_CONCEPT.md).

| File | Content |
|---|---|
| `01_data_analysis.ipynb` | preliminary data analysis: data quality, price drivers, where the savings are |
| `02_ml_model.ipynb` | features, five models, walk-forward evaluation, error analysis, hold-out |
| `ml_model.py` | daily run: trains the final LightGBM on the latest 365 days and appends D+2 / D+3 predictions to `predictions` |

**Approach:** the model predicts all 24 hourly prices of D+2 and D+3 as the change vs. the last known day (D+1), from weather forecasts (wind and solar proxies, heating degrees), recent price shapes and the calendar. Prices are scaled robustly (asinh/MAD); all fitted parts are refitted monthly on the previous 365 days (walk-forward), so no test data leaks into training.

**Results** (MAE in Rp./kWh in the fleet's charging hours; improvement over the better simple rule, "copy of D+1" or "last week"):

| | Development (Mar 2025 – May 2026) | Hold-out (May – Nov 2026, run once) |
|---|---|---|
| D+2, LightGBM | 0.87 (+26 %) | 1.48 (+8 %) |
| D+3, LightGBM | 1.05 (+30 %) | 1.79 (+22 %) |

The hold-out includes a doubling of the price level. Relative to the price level the error stays at about 9 % (D+2) and 11 % (D+3); the lead over copying the known day shrinks because that period was driven more by a persistent price level than by weather. Known weak spots: sunny spring Sundays and holidays with negative prices, and sudden level jumps. Details and possible improvements: `02_ml_model.ipynb`.

## Part 3: App

*To be written by the app team.*

## Team contributions

*Contribution matrix, see video.*

## Use of generative AI

The code in `BuildDB_v2.py`, `ml_model.py`, `01_data_analysis.ipynb` and `02_ml_model.ipynb` was created with the help of Claude (Anthropic) and was reviewed and adapted by the team.
