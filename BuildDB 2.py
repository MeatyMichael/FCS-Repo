"""
BuildDB.py
----------
Builds or updates the SQLite database "electricity_prices.db".

The app plans when a company's electric vehicles can charge most cheaply. For
that it needs electricity prices for the next few days. The exchange, however,
only publishes prices up to tomorrow. The day after tomorrow and the day after
that are predicted by an ML model. This database provides the model's training
data and stores everything the app works with.

Market data (updated on every run, older values are reused):
  prices_ch          Day-ahead price Switzerland per hour             (Energy-Charts API)
  prices_de          Day-ahead price Germany/Luxembourg (DE-LU)       (Energy-Charts API)
  weather            measured weather: avg. of Swiss cities + wind at 3 locations
                     in northern Germany                              (Open-Meteo API)
  weather_forecasts  weather as it was forecast 2 or 3 days in advance
                                                       (Open-Meteo Previous Runs API)
  ml_data            prices + weather + features for ML training
App tables (filled with sample data on the first run only, never overwritten afterwards):
  tariffs, vehicle_types, vehicles, schedule, settings, predictions, charging_plan

Run after approx. 13:00. Before that, tomorrow's prices are not yet published
and the "day after tomorrow" model is missing its most important feature.
The app can also do this automatically (see guide 2 at the very bottom).

Data sources: api.energy-charts.info (Fraunhofer ISE) and open-meteo.com, both CC BY 4.0
AI notice: Created with Claude (Anthropic), reviewed and adapted by us.

Installation:   pip install -r requirements.txt
Run:            python BuildDB.py
"""

import sqlite3
import threading
import time

import holidays
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DATABASE = "electricity_prices.db"

# From 2024: one full year for training plus test months, so that the model
# sees every season at least once.
START = "2024-01-01"

EUR_TO_CHF = 0.94                          # assumption: fixed exchange rate (API returns euros)

# SQLite has no dedicated date type, so we store times as text in a fixed,
# sortable format. All times in the DB are in UTC.
TIME_FORMAT = "%Y-%m-%d %H:%M"

# The exchange price applies to all of Switzerland. Weather at a single location
# would be too random, so we use the average of several regions.
WEATHER_LOCATIONS = {
    "Zurich": (47.37, 8.54),
    "Bern": (46.95, 7.45),
    "Geneva": (46.20, 6.14),
    "Basel": (47.56, 7.59),
    "Lugano": (46.00, 8.95),
}

# The large wind farms are in northern Germany. Strong wind there pushes prices
# down across central Europe, including Switzerland. These locations are NOT
# averaged but stored as separate columns: the model decides for itself how
# much weight to give each region. We only use the wind from here.
# column name: (latitude, longitude)
WIND_LOCATIONS_DE = {
    "wind_de_hamburg": (53.55, 9.99),      # Hamburg
    "wind_de_husum": (54.48, 9.05),        # Husum (North Sea coast)
    "wind_de_emden": (53.37, 7.21),        # Emden (East Frisia)
}

# Weather values that influence the price:
#   Temperature: cold -> more heating -> more demand -> higher price
#   Wind:        lots of wind -> lots of wind power -> lower price. 100 m instead
#                of 10 m, because wind turbines stand at roughly that height.
#   Solar:       lots of sun -> lots of solar power -> lower midday prices
WEATHER_VARIABLES = {
    "temperature_2m": "temperature",       # °C
    "wind_speed_100m": "wind_100m",        # m/s (see "wind_speed_unit" below)
    "shortwave_radiation": "solar",        # W/m²
}
WIND_VARIABLE = "wind_speed_100m"          # the only variable we use for northern Germany

# All measured weather columns in ml_data
WEATHER_COLUMNS = list(WEATHER_VARIABLES.values()) + list(WIND_LOCATIONS_DE)

# Pass with ALL Open-Meteo requests: wind consistently in m/s instead of km/h.
# This avoids silent errors when sources are mixed.
OPEN_METEO_PARAMS = {"wind_speed_unit": "ms", "timezone": "GMT"}  # GMT = times in UTC

# Weather forecasts for training (Previous Runs API).
# A fixed model instead of "best match", so the source cannot change unnoticed.
# Checked on 24.11.2026: in the Previous Runs API, ecmwf_ifs025 provides
# temperature from 5.2.2024, wind at 100 m and radiation only from 9.3.2024.
# Before that the _fc columns stay empty (the ML script drops those rows).
WEATHER_MODEL = "ecmwf_ifs025"

# How many days before the target hour the forecast was made, per column suffix.
# Known inaccuracy: "2 days before" = 48 h before the target hour. For day 2 at
# 18:00 that is day 0 at 18:00, i.e. slightly AFTER the decision at 13:00.
# For weather forecasts the difference is small. To be strict, set
# {"fc_d2": 3, "fc_d3": 4}.
FORECAST_DAYS_AHEAD = {"fc_d2": 2, "fc_d3": 3}

# Bidding zones for electricity prices and the table they are stored in.
# Swiss prices closely follow German ones (strongly coupled markets).
PRICE_TABLES = {"CH": "prices_ch", "DE-LU": "prices_de"}

# The APIs block us if we make too many requests ("429 Too Many Requests").
# Therefore: few large requests, pauses in between, and reuse already stored
# values instead of reloading everything each time.
PRICE_BLOCK_DAYS = 90                      # days per price request
PAUSE_SECONDS = 10                         # pause between requests (Energy-Charts allows only a few per minute)
RELOAD_DAYS = 14                           # always reload the last days in case values were corrected


def today() -> pd.Timestamp:
    """
    Today's date (midnight, Swiss time).
    A function instead of a fixed variable, because the app can run for days.
    A variable would only be computed at startup and be wrong the next day.
    """
    return pd.Timestamp.now(tz="Europe/Zurich").normalize()


def fetch_json(url: str, params: dict, timeout: int = 60):
    """
    Queries an API and returns the response as JSON.
    If we still hit the limit (429), the function waits and tries again instead
    of aborting the whole script. Other errors abort immediately, so that we do
    not continue calculating with wrong data.
    """
    for attempt in range(5):
        response = requests.get(url, params=params, timeout=timeout)
        if response.status_code != 429:
            response.raise_for_status()    # other errors: abort
            return response.json()
        retry_after = response.headers.get("Retry-After", "")
        wait = int(retry_after) if retry_after.isdigit() else 30 * 2 ** attempt
        print(f"    API limit reached, waiting {wait} s ...")
        time.sleep(wait)
    response.raise_for_status()


# ---------------------------------------------------------------------------
# 1. Electricity prices
# The price is what the model should predict (label). Earlier prices also
# serve as a hint about future ones (features).
# ---------------------------------------------------------------------------
def load_stored_table(table: str) -> pd.DataFrame:
    """
    Reads a market data table that is already in the database (empty on the
    first run). Old values no longer change. By reusing them, a daily run only
    needs one small request per source.
    """
    try:
        with sqlite3.connect(DATABASE) as connection:
            old = pd.read_sql(f"SELECT * FROM {table}", connection)
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()
    old["time_utc"] = pd.to_datetime(old["time_utc"], format=TIME_FORMAT, utc=True)
    return old


def reload_start(old: pd.DataFrame) -> pd.Timestamp:
    """From when to reload: everything on the first run, otherwise the last 14 days."""
    start = pd.Timestamp(START)
    if not old.empty:
        last = old["time_utc"].max().tz_convert(None).normalize()
        start = max(start, last - pd.Timedelta(days=RELOAD_DAYS))
        print(f"  {len(old)} hours from the database, reloading from {start:%Y-%m-%d}")
    return start


def load_prices(zone: str = "CH") -> pd.DataFrame:
    """
    Day-ahead prices of a bidding zone ("CH" or "DE-LU") up to and including
    tomorrow, one row per hour. Only loads what is missing from the database,
    plus the last 14 days. Each zone has its own table (PRICE_TABLES).
    Returns: time_utc, price_rp_kwh
    """
    # Up to tomorrow, because these prices are already known (from approx. 13:00).
    # They are the most important feature for the day-after-tomorrow forecast.
    end = (today() + pd.Timedelta(days=1)).tz_localize(None)

    old = load_stored_table(PRICE_TABLES[zone])
    start = reload_start(old)

    # Query in blocks of PRICE_BLOCK_DAYS
    parts = []
    block_start = start
    while block_start <= end:
        block_end = min(block_start + pd.Timedelta(days=PRICE_BLOCK_DAYS - 1), end)
        print(f"  Prices {zone} {block_start:%Y-%m-%d} to {block_end:%Y-%m-%d} ...")
        data = fetch_json("https://api.energy-charts.info/price",
                          {"bzn": zone, "start": block_start.strftime("%Y-%m-%d"),
                           "end": block_end.strftime("%Y-%m-%d")})
        parts.append(pd.DataFrame({
            "time_utc": pd.to_datetime(data["unix_seconds"], unit="s", utc=True),
            "price_eur_mwh": data["price"],
        }))
        block_start = block_end + pd.Timedelta(days=1)
        if block_start <= end:
            time.sleep(PAUSE_SECONDS)

    new = pd.concat(parts).dropna().drop_duplicates("time_utc")

    # Since October 2025 there are prices per 15 minutes. Model and app work in
    # hours, so we average the four values of an hour.
    new = new.set_index("time_utc").resample("h").mean().dropna().reset_index()

    # Rp./kWh is the unit on every Swiss electricity bill.
    # 1 MWh = 1000 kWh and 1 EUR = 100 cents -> divide by 10, then convert to francs.
    new["price_rp_kwh"] = (new["price_eur_mwh"] / 10 * EUR_TO_CHF).round(3)
    new = new[["time_utc", "price_rp_kwh"]]

    # Merge old + new, on overlap the new value wins
    prices = new if old.empty else pd.concat([old, new]).drop_duplicates("time_utc", keep="last")
    return prices.sort_values("time_utc").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Weather
# Wind, sun and temperature influence how much electricity is produced and
# consumed, and therefore the price.
# ---------------------------------------------------------------------------
def fetch_weather_all_locations(url: str, hourly: list[str], params: dict) -> pd.DataFrame:
    """
    Loads the weather for all locations (Switzerland + northern Germany) with a
    single request. Open-Meteo allows several locations per request
    (comma-separated coordinates). One request instead of eight spares the API limit.
    The API requests the same variables for all locations. That costs nothing
    extra (Open-Meteo only counts double from 10 variables), but for northern
    Germany only the wind is used afterwards (see combine_locations).
    Returns: one row per location and hour, columns "location", "time_utc" + hourly
    """
    locations = {**WEATHER_LOCATIONS, **WIND_LOCATIONS_DE}
    latitudes = ",".join(str(lat) for lat, _ in locations.values())
    longitudes = ",".join(str(lon) for _, lon in locations.values())
    data = fetch_json(url, {"latitude": latitudes, "longitude": longitudes,
                            "hourly": ",".join(hourly), **OPEN_METEO_PARAMS, **params},
                      timeout=180)
    if isinstance(data, dict):             # with only one location, no list is returned
        data = [data]
    parts = []
    for name, location in zip(locations, data):  # responses come in the same order
        table = pd.DataFrame(location["hourly"])
        table["time_utc"] = pd.to_datetime(table.pop("time"), utc=True)
        table["location"] = name
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


def combine_locations(raw: pd.DataFrame, api_suffix: str = "", suffix: str = "") -> pd.DataFrame:
    """
    Turns one row per location and hour into one row per hour:
      Swiss cities:      average -> temperature, wind_100m, solar
      Northern Germany:  wind only, one column per location -> wind_de_hamburg, ...
    api_suffix: suffix of the API variables (e.g. "_previous_day2")
    suffix:     suffix of our columns (e.g. "_fc_d2")
    """
    names = {api + api_suffix: column + suffix for api, column in WEATHER_VARIABLES.items()}
    switzerland = raw[raw["location"].isin(WEATHER_LOCATIONS)]
    result = switzerland.groupby("time_utc")[list(names)].mean().rename(columns=names)

    germany = raw[raw["location"].isin(WIND_LOCATIONS_DE)]
    wind_de = germany.pivot(index="time_utc", columns="location", values=WIND_VARIABLE + api_suffix)
    wind_de = wind_de[list(WIND_LOCATIONS_DE)].add_suffix(suffix)  # fixed column order
    return result.join(wind_de, how="outer")


def load_weather() -> pd.DataFrame:
    """
    Hourly measured weather, from two sources:
      Archive  = measured values for training (approx. 5 days behind)
      Forecast = fills the gap to the archive and the next 3 days
    For training we need the past, for predicting the weather forecast for the
    day after tomorrow and the day after that.
    Returns: time_utc, temperature, wind_100m, solar, wind_de_*, is_forecast
    """
    print("  Weather archive (all locations) ...")
    archive = fetch_weather_all_locations(
        "https://archive-api.open-meteo.com/v1/archive", list(WEATHER_VARIABLES),
        {"start_date": START, "end_date": (today() - pd.Timedelta(days=6)).strftime("%Y-%m-%d")})
    time.sleep(PAUSE_SECONDS)
    print("  Weather forecast (all locations) ...")
    forecast = fetch_weather_all_locations(
        "https://api.open-meteo.com/v1/forecast", list(WEATHER_VARIABLES),
        {"past_days": 7, "forecast_days": 4})

    archive = combine_locations(archive).dropna()
    forecast = combine_locations(forecast).dropna()
    # Measured values are more accurate, so the archive takes precedence. The
    # forecast only fills the hours still missing from the archive.
    forecast = forecast[forecast.index > archive.index.max()]
    weather = pd.concat([archive, forecast]).round(2).rename_axis("time_utc").reset_index()

    # Marks hours whose weather is only forecast. Important for evaluation:
    # training contains measured weather, real predictions only a forecast.
    weather["is_forecast"] = (weather["time_utc"] > pd.Timestamp.now(tz="UTC")).astype(int)
    return weather


def load_weather_forecasts() -> pd.DataFrame:
    """
    Weather as it was forecast 2 or 3 days in advance (Previous Runs API).
    Why? At 13:00 on day 0, only a forecast is known for day 2 and day 3. If the
    model trains on measured weather, it learns with better information than it
    gets in practice and looks better in the test than it really is.
    Incremental like the prices: reuse stored values, only reload the last
    14 days and the future. The request reaches up to day 3, because for
    upcoming hours the API returns the current forecast.
    Returns: time_utc + for each weather column the variants _fc_d2 and _fc_d3
    """
    old = load_stored_table("weather_forecasts")
    start = reload_start(old)
    end = today() + pd.Timedelta(days=3)

    hourly = [f"{v}_previous_day{days}"
              for v in WEATHER_VARIABLES for days in sorted(set(FORECAST_DAYS_AHEAD.values()))]
    print(f"  Weather forecasts {WEATHER_MODEL} {start:%Y-%m-%d} to {end:%Y-%m-%d} (all locations) ...")
    raw = fetch_weather_all_locations(
        "https://previous-runs-api.open-meteo.com/v1/forecast", hourly,
        {"start_date": start.strftime("%Y-%m-%d"), "end_date": end.strftime("%Y-%m-%d"),
         "models": WEATHER_MODEL})

    parts = [combine_locations(raw, f"_previous_day{days}", f"_{suffix}")
             for suffix, days in FORECAST_DAYS_AHEAD.items()]
    new = pd.concat(parts, axis=1).round(2).rename_axis("time_utc").reset_index()

    # Merge old + new, on overlap the new value wins
    forecasts = new if old.empty else pd.concat([old, new]).drop_duplicates("time_utc", keep="last")
    return forecasts.sort_values("time_utc").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. ML data
# Everything the model needs, in one table. The ML script only has to read
# this table.
# ---------------------------------------------------------------------------
def build_ml_data(prices: pd.DataFrame, prices_de: pd.DataFrame,
                  weather: pd.DataFrame, weather_forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    One row per hour, from the start until 3 days into the future.
      Rows with a price    -> training and test
      Rows without a price -> these hours are predicted by the model

    Two models, so that both use the known price of tomorrow:
      "day after tomorrow":  price_lag_24h + price_lag_168h
      "day after that":      price_lag_48h + price_lag_168h
    The "day after that" model must NOT use price_lag_24h: that would be the price
    of the day after tomorrow, which is not yet known at prediction time. In
    training the column would still be filled, and the model would look too good
    (data leakage).
    """
    # The empty future rows are exactly the hours for which the app needs a
    # prediction.
    all_hours = pd.date_range(prices["time_utc"].min(),
                              today().tz_convert("UTC") + pd.Timedelta(days=4),
                              freq="h", inclusive="left")
    data = pd.DataFrame({"time_utc": all_hours})
    data = (data.merge(prices, on="time_utc", how="left")
                .merge(weather, on="time_utc", how="left")
                .merge(weather_forecasts, on="time_utc", how="left"))

    # Future rows: if the Previous Runs API returns nothing there (yet), fill
    # with the current weather forecast. That is correct, because this forecast
    # really is 2 or 3 days ahead. Past gaps stay empty, otherwise measured
    # weather would end up there (data leakage).
    future = data["time_utc"] > pd.Timestamp.now(tz="UTC")
    for column in WEATHER_COLUMNS:
        for suffix in FORECAST_DAYS_AHEAD:
            fc = f"{column}_{suffix}"
            data.loc[future, fc] = data.loc[future, fc].fillna(data.loc[future, column])

    # Calendar features in Swiss time, because the daily rhythm (getting up,
    # working, cooking) follows the local clock.
    local = data["time_utc"].dt.tz_convert("Europe/Zurich")
    data["time_local"] = local.dt.strftime(TIME_FORMAT)
    data["hour"] = local.dt.hour                      # cheap at night, expensive morning/evening
    data["weekday"] = local.dt.day_of_week            # 0 = Monday
    data["weekend"] = (data["weekday"] >= 5).astype(int)  # less industry/offices
    data["month"] = local.dt.month                    # winter more expensive

    # On public holidays demand is as low as on a Sunday. Canton of Zurich,
    # because many holidays (Good Friday, St. Stephen's Day ...) are cantonal.
    holidays_ch = holidays.country_holidays("CH", subdiv="ZH", years=range(2024, today().year + 2))
    data["holiday_ch"] = local.dt.date.map(lambda day: int(day in holidays_ch))

    # German holidays: demand drops there, and that also moves Swiss prices.
    # Without subdiv = nationwide holidays only. Assigned like above by the
    # date in Swiss time.
    holidays_de = holidays.country_holidays("DE", years=range(2024, today().year + 2))
    data["holiday_de"] = local.dt.date.map(lambda day: int(day in holidays_de))

    # Earlier prices as features. Electricity prices rarely change abruptly;
    # the price at the same hour yesterday/last week is a strong hint.
    #   24 h / 48 h: tomorrow's price for the "day after tomorrow" / "day after that" models
    #   168 h:       same hour one week earlier (weekly rhythm)
    # Implementation: shift the price table X hours forward and merge it.
    for hours in (24, 48, 168):
        earlier = prices.assign(time_utc=prices["time_utc"] + pd.Timedelta(hours=hours))
        data = data.merge(earlier.rename(columns={"price_rp_kwh": f"price_lag_{hours}h"}),
                          on="time_utc", how="left")

    # German prices using the same principle. Do NOT store the German price of
    # the same hour (without shift): for day 2 and day 3 it is still unknown
    # (data leakage).
    for hours in (24, 48, 168):
        earlier = prices_de.assign(time_utc=prices_de["time_utc"] + pd.Timedelta(hours=hours))
        data = data.merge(earlier.rename(columns={"price_rp_kwh": f"price_de_lag_{hours}h"}),
                          on="time_utc", how="left")

    forecast_columns = [f"{column}_{suffix}"
                        for suffix in FORECAST_DAYS_AHEAD for column in WEATHER_COLUMNS]
    data["time_utc"] = data["time_utc"].dt.strftime(TIME_FORMAT)
    return data[["time_utc", "time_local", "price_rp_kwh",
                 "hour", "weekday", "weekend", "holiday_ch", "holiday_de", "month",
                 *WEATHER_COLUMNS, "is_forecast",
                 "price_lag_24h", "price_lag_48h", "price_lag_168h",
                 "price_de_lag_24h", "price_de_lag_48h", "price_de_lag_168h",
                 *forecast_columns]]


# ---------------------------------------------------------------------------
# 4. App tables
# The app has to remember which vehicles exist, when they drive, which tariff
# applies and which charging plans and predictions have been calculated.
# ---------------------------------------------------------------------------
APP_TABLES = """
-- ML predictions with creation time (for comparing prediction vs. actual price)
CREATE TABLE IF NOT EXISTS predictions (
    created_at          TEXT NOT NULL,
    time_utc            TEXT NOT NULL,     -- predicted hour
    predicted_rp_kwh    REAL NOT NULL,
    PRIMARY KEY (created_at, time_utc)
);

-- The exchange price is the same across Switzerland, but the customer also pays
-- grid fees, levies and the local utility's margin: final price = exchange price + surcharge.
-- With peak/off-peak tariffs the charging plan can change as well.
CREATE TABLE IF NOT EXISTS tariffs (
    tariff_id              INTEGER PRIMARY KEY,
    name                   TEXT NOT NULL,
    surcharge_rp_kwh       REAL NOT NULL,
    peak_surcharge_rp_kwh  REAL,          -- surcharge during peak tariff
    peak_start             INTEGER,       -- peak tariff from hour
    peak_end               INTEGER,       -- peak tariff until hour (exclusive)
    source                 TEXT
);

-- Technical data per vehicle model
CREATE TABLE IF NOT EXISTS vehicle_types (
    type_id                INTEGER PRIMARY KEY,
    name                   TEXT NOT NULL,
    battery_kwh            REAL NOT NULL,
    consumption_kwh_100km  REAL NOT NULL,
    charging_power_kw      REAL NOT NULL
);

-- Fleet. Arrival/departure determine WHEN charging is possible,
-- battery level and reserve determine HOW MUCH must be charged at least.
CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_id           INTEGER PRIMARY KEY,
    name                 TEXT NOT NULL,
    type_id              INTEGER NOT NULL REFERENCES vehicle_types(type_id),
    depot_arrival        TEXT NOT NULL,   -- "18:00"
    depot_departure      TEXT NOT NULL,   -- "07:00"
    battery_percent      REAL NOT NULL,
    min_reserve_percent  REAL NOT NULL DEFAULT 30
);

-- km per weekday. From this the app calculates how much electricity a vehicle
-- needs. 0 km = stays in the depot all day and can charge (e.g. Sunday).
CREATE TABLE IF NOT EXISTS schedule (
    vehicle_id  INTEGER NOT NULL REFERENCES vehicles(vehicle_id) ON DELETE CASCADE,
    weekday     INTEGER NOT NULL CHECK (weekday BETWEEN 0 AND 6),  -- 0 = Monday
    km          REAL NOT NULL,
    PRIMARY KEY (vehicle_id, weekday)
);

-- Settings the user changes in the app
CREATE TABLE IF NOT EXISTS settings (
    name   TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);

-- Saved charging plans. These produce the "currently charging" display.
CREATE TABLE IF NOT EXISTS charging_plan (
    created_at  TEXT NOT NULL,
    vehicle_id  INTEGER NOT NULL REFERENCES vehicles(vehicle_id) ON DELETE CASCADE,
    time_utc    TEXT NOT NULL,
    kwh         REAL NOT NULL,
    PRIMARY KEY (created_at, vehicle_id, time_utc)
);
"""

# Sample data (parcel service), so that the app is not empty when first opened
# and you can immediately see how it works.
# Tariff surcharges are PLACEHOLDERS -> look up real values at ElCom
SAMPLE_TARIFFS = [
    ("Exchange price only", 0.0, None, None, None, "-"),
    ("Example: fixed surcharge", 15.0, None, None, None, "PLACEHOLDER"),
    ("Example: peak/off-peak", 12.0, 18.0, 7, 20, "PLACEHOLDER"),
]
# name, battery_kwh, consumption_kwh_100km, charging_power_kw (guide values)
SAMPLE_VEHICLE_TYPES = [
    ("Small electric van", 45, 20, 11),
    ("Large electric transporter", 80, 28, 11),
    ("Electric tricycle", 7, 8, 3.7),
]
# name, type_id, arrival, departure, battery %, reserve %, km Mon..Sun
# Sunday 0 km (no deliveries) -> the app can fully charge cheaply there.
SAMPLE_FLEET = [
    ("Van 1", 1, "18:00", "07:00", 55, 30, [70, 60, 60, 60, 65, 35, 0]),
    ("Van 2", 1, "18:30", "07:30", 40, 30, [80, 70, 70, 70, 75, 40, 0]),
    ("Transporter 1", 2, "17:30", "06:30", 60, 30, [110, 95, 95, 95, 100, 50, 0]),
    ("Tricycle 1", 3, "16:00", "08:00", 50, 25, [30, 25, 25, 25, 28, 15, 0]),
]
SAMPLE_SETTINGS = [
    ("num_charging_stations", "4"),
    ("selected_tariff", "2"),
]


def create_app_tables(connection) -> None:
    """
    Creates the app tables and fills them with sample data only the first time.
    The script runs daily. User input (own vehicles, tariffs) must not be
    overwritten in the process.
    """
    connection.executescript(APP_TABLES)   # "IF NOT EXISTS": only create if missing
    cur = connection.cursor()

    if cur.execute("SELECT COUNT(*) FROM tariffs").fetchone()[0] > 0:
        return                             # already filled -> don't touch user input

    cur.executemany("""INSERT INTO tariffs (name, surcharge_rp_kwh, peak_surcharge_rp_kwh,
                       peak_start, peak_end, source) VALUES (?, ?, ?, ?, ?, ?)""", SAMPLE_TARIFFS)
    cur.executemany("""INSERT INTO vehicle_types (name, battery_kwh, consumption_kwh_100km,
                       charging_power_kw) VALUES (?, ?, ?, ?)""", SAMPLE_VEHICLE_TYPES)
    for name, type_id, arrival, departure, battery, reserve, km_per_day in SAMPLE_FLEET:
        cur.execute("""INSERT INTO vehicles (name, type_id, depot_arrival, depot_departure,
                       battery_percent, min_reserve_percent) VALUES (?, ?, ?, ?, ?, ?)""",
                    (name, type_id, arrival, departure, battery, reserve))
        vehicle_id = cur.lastrowid         # ID of the vehicle just created, for the schedule
        cur.executemany("INSERT INTO schedule VALUES (?, ?, ?)",
                        [(vehicle_id, day, km) for day, km in enumerate(km_per_day)])
    cur.executemany("INSERT INTO settings VALUES (?, ?)", SAMPLE_SETTINGS)


# ---------------------------------------------------------------------------
# 5. Write database
# ---------------------------------------------------------------------------
def build_database() -> None:
    """Loads the market data and writes everything to the database (creates it if needed)."""
    print("1/5 Loading electricity prices ...")
    prices = load_prices("CH")
    time.sleep(PAUSE_SECONDS)
    prices_de = load_prices("DE-LU")
    print("2/5 Loading weather ...")
    weather = load_weather()
    time.sleep(PAUSE_SECONDS)
    print("3/5 Loading weather forecasts for training ...")
    weather_forecasts = load_weather_forecasts()
    print("4/5 Building ML data ...")
    ml_data = build_ml_data(prices, prices_de, weather, weather_forecasts)

    print("5/5 Writing database ...")
    with sqlite3.connect(DATABASE) as connection:
        # SQLite only checks references if you turn it on. This way, e.g. no
        # schedule is left over for a deleted vehicle.
        connection.execute("PRAGMA foreign_keys = ON")

        # Replace market data tables completely. This is safe, because "prices"
        # already contains the old prices from the database.
        for table, df in [("prices_ch", prices), ("prices_de", prices_de),
                          ("weather", weather), ("weather_forecasts", weather_forecasts)]:
            df.assign(time_utc=df["time_utc"].dt.strftime(TIME_FORMAT)).to_sql(
                table, connection, if_exists="replace", index=False)
        ml_data.to_sql("ml_data", connection, if_exists="replace", index=False)
        create_app_tables(connection)

    print(f"Done: {DATABASE} ({len(ml_data)} hours in ml_data)")


# ---------------------------------------------------------------------------
# 6. Automatic update
# The app calls update_if_needed() when it opens. This way nobody has to
# remember to start this script by hand every day.
# ---------------------------------------------------------------------------
PRICES_PUBLISHED_AT = 13                   # from then on, tomorrow's prices are available
RETRY_INTERVAL_MIN = 30                    # minimum time between two attempts

_last_attempt = None                       # time of the last attempt (while the app is running)
_lock = threading.Lock()                   # in case several users open the app at the same time


def is_up_to_date() -> bool:
    """
    Checks whether the database contains all prices that should already exist:
      before 13:00: until end of today
      from 13:00:   until end of tomorrow (then tomorrow's prices are published)
    Fast (only one SQL query), so the app can check this on every load.
    """
    now = pd.Timestamp.now(tz="Europe/Zurich")
    days = 2 if now.hour >= PRICES_PUBLISHED_AT else 1
    # last hour that must be present (e.g. tomorrow 23:00 Swiss time), in UTC
    needed_until = (today() + pd.Timedelta(days=days) - pd.Timedelta(hours=1)).tz_convert("UTC")
    try:
        with sqlite3.connect(DATABASE) as connection:
            last = connection.execute("SELECT MAX(time_utc) FROM prices_ch").fetchone()[0]
    except sqlite3.Error:
        return False                       # database or table missing
    return last is not None and last >= needed_until.strftime(TIME_FORMAT)


def update_if_needed() -> str | None:
    """
    Updates the database if it is not up to date.
    Returns: None if everything is up to date, otherwise a notice for the app.

    Never raises an error (e.g. without internet), so that the app keeps running
    with the old data instead of crashing.
    Tries at most every 30 minutes: sometimes the prices only appear after
    13:00, and otherwise every click in the app would trigger an API request.
    """
    global _last_attempt
    if is_up_to_date():
        return None
    with _lock:
        if is_up_to_date():                # updated by someone else in the meantime
            return None
        now = pd.Timestamp.now(tz="Europe/Zurich")
        if (_last_attempt is not None
                and now - _last_attempt < pd.Timedelta(minutes=RETRY_INTERVAL_MIN)):
            return "The latest electricity prices are not available yet. Using the last known data."
        _last_attempt = now
        try:
            build_database()
        except Exception as error:         # deliberately broad: the app must never crash here
            return f"Update failed ({error}). Using the last known data."
    if not is_up_to_date():
        return "Tomorrow's electricity prices have not been published yet. Next attempt in 30 minutes."
    return None


# Start by hand: python BuildDB.py
if __name__ == "__main__":
    build_database()


# ===========================================================================
# GUIDE 1: ML MODEL  (file still to be created, e.g. ml_model.py)
# ===========================================================================
#
# GOAL
#   Predict the exchange price for the day after tomorrow and the day after
#   that, and write it to the "predictions" table. As long as it is empty, the
#   app uses the price of the previous week as a placeholder.
#
# INPUT: table "ml_data" (one row per hour, times in UTC)
#   Label:     price_rp_kwh
#   Features:  hour, weekday, weekend, holiday_ch, holiday_de, month,
#              price_lag_24h, price_lag_48h, price_lag_168h,
#              price_de_lag_24h, price_de_lag_48h, price_de_lag_168h,
#              weather as forecast (see below)
#   Measured weather (only for the comparison "measured vs. forecast"):
#              temperature, wind_100m (m/s), solar,
#              wind_de_hamburg, wind_de_husum, wind_de_emden (m/s)
#   Weather as forecast: the same columns with suffix _fc_d2 (forecast 2 days
#              in advance) or _fc_d3 (3 days in advance), e.g.
#              temperature_fc_d2, wind_de_hamburg_fc_d3.
#              Model ecmwf_ifs025, complete only from approx. 9.3.2024.
#   Not features: time_utc, time_local, is_forecast
#
# TWO MODELS (because of data leakage)
#   "day after tomorrow":  without price_lag_48h / price_de_lag_48h, weather *_fc_d2
#   "day after that":      without price_lag_24h / price_de_lag_24h, weather *_fc_d3
#
# TRAINING
#   - Only rows with price_rp_kwh, drop rows with missing features.
#   - Split by time, not randomly (e.g. last 3 months = test).
#   - Compare with baseline "price = price_lag_168h" (MAE in Rp./kWh).
#     The model must be better.
#   - Suggestion: sklearn RandomForestRegressor or HistGradientBoostingRegressor
#     (add scikit-learn to requirements.txt).
#
# PREDICTION: rows without price_rp_kwh
#   price_lag_24h present                  -> "day after tomorrow" model
#   price_lag_24h empty, _48h present      -> "day after that" model
#   With the _fc columns, the weather in training and prediction is the same
#   kind (both forecasts), so the test is honest.
#
# OUTPUT: table "predictions"
#   created_at (text, TIME_FORMAT), time_utc (as in ml_data), predicted_rp_kwh
#   Append (INSERT OR REPLACE / if_exists="append"), NEVER "replace":
#   old predictions are needed by the "Prediction quality" tab.
#
# PROCESS: run daily after BuildDB.py.
#
#
# ===========================================================================
# GUIDE 2: APP
# ===========================================================================
#
# INSTALLATION
#   pip install -r requirements.txt
#
# START
#   python -m streamlit run app.py
#   ("python -m", because streamlit.exe is often not on the PATH on Windows)
#   All files in the same folder as electricity_prices.db.
#
# AUTOMATIC UPDATE
#   The app can keep the database up to date by itself. Then nobody has to
#   start BuildDB.py by hand. In app.py, BEFORE the "database not found"
#   check, insert:
#
#       import BuildDB
#
#       with st.spinner("Checking whether new electricity prices are available ..."):
#           notice = BuildDB.update_if_needed()
#       if notice:
#           st.info(notice)
#
#   Behaviour:
#   - Database up to date: only one check (approx. 0.01 s), no API request.
#   - Prices missing: update within a few seconds.
#   - Database missing: gets created (2-3 minutes the first time).
#   - No internet / prices not yet published: app keeps running with old
#     data, shows a notice, next attempt after 30 min at the earliest.
#   - Several users at the same time: only one update runs.
#   - "Up to date" means: before 13:00 prices until end of today, after that
#     until end of tomorrow.
#   The ML predictions are NOT updated in the process (run ml_model.py separately).
#
# STRUCTURE
#   app.py          user interface (Streamlit)
#   ladeplanung.py  calculation logic: prices, final price with tariff, charging plan
#
# TABS
#   Charging plan       when which vehicle charges
#   Electricity prices  prices of the next days with source (exchange / ML / placeholder)
#   Fleet               edit vehicles and schedules
#   Prediction quality  ML prediction vs. actual price (once predictions exist)
#   Tariffs             edit surcharges (sample values are placeholders)
#
# IMPORTANT
#   User input is kept on every update. Only deleting electricity_prices.db
#   resets everything to the sample data.
