"""
daten.py - everything the pages read from or write to electricity_prices.db (role A, Data).

electricity_prices.db is the SQLite database that BuildDB.py (A) builds: real
day-ahead prices for Switzerland and Germany (Energy-Charts API), weather and
weather forecasts (Open-Meteo API), the ML table and the app tables (tariffs,
vehicle types, vehicles, km per weekday, settings, predictions, charging plans).
Run `python BuildDB.py` once (a few minutes) to download the prices.
Without the prices the app still opens: the fleet and tariff tables are then
created with BuildDB.py's sample data, and the prices are made up (the
sidebar says so).

Generated with Claude (Anthropic): made-up prices on 19 Nov 2026, the
database version on 23 Nov 2026, English table names on 27 Nov 2026.
A reviews this file and owns it.

What the pages use:
    get_prices(tariff_id, start, end) -> DataFrame ts, eur_mwh, rp_kwh, is_forecast
    get_market_prices(start, end)    -> DataFrame ts, market_rp, is_forecast
    get_tariffs()                    -> DataFrame from the table tariffs
    get_vehicle_types()              -> DataFrame from the table vehicle_types
    get_fleet() / save_fleet(fleet, types) -> the vans with their km per weekday
    get_setting(key, default) / save_setting(key, value)
    save_charging_plan(blocks_of_van)  -> writes the table charging_plan
    published_until(), data_source(), get_meter(typical_kwh, days)
"""

import os
import sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import BuildDB  # A's script: table definitions and sample data

TZ = ZoneInfo("Europe/Zurich")

# Same file name, time format and exchange rate as in BuildDB.py
DATABASE = BuildDB.DATABASE
TIME_FORMAT = BuildDB.TIME_FORMAT   # times are stored as UTC text, e.g. "2026-11-23 17:00"
EUR_CHF = BuildDB.EUR_TO_CHF

# One fleet column per weekday (km driven), in the order of the table
# schedule: weekday 0 = Monday ... 6 = Sunday
KM_COLUMNS = ["km_mon", "km_tue", "km_wed", "km_thu", "km_fri", "km_sat", "km_sun"]


# ---------------------------------------------------------------------------
# 1. Connection
# ---------------------------------------------------------------------------
def _connect():
    """Open electricity_prices.db and make sure the app tables exist.

    BuildDB.create_app_tables() creates the tables if they are missing
    and fills them with sample data only the very first time. It never
    overwrites what the user has entered.
    """
    connection = sqlite3.connect(DATABASE)
    BuildDB.create_app_tables(connection)
    connection.commit()
    return connection


def get_setting(key, default):
    """Read one value from the table settings (always text)."""
    connection = _connect()
    row = connection.execute("SELECT value FROM settings WHERE name = ?", (key,)).fetchone()
    connection.close()
    return default if row is None else row[0]


def save_setting(key, value):
    """Store one value in the table settings."""
    connection = _connect()
    connection.execute("INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, str(value)))
    connection.commit()
    connection.close()


# ---------------------------------------------------------------------------
# 2. Tariffs, vehicle types and the fleet
# ---------------------------------------------------------------------------
def get_tariffs():
    """All tariffs: name, surcharge on the market price, optional peak tariff."""
    connection = _connect()
    tariffs = pd.read_sql("SELECT * FROM tariffs ORDER BY tariff_id", connection)
    connection.close()
    return tariffs


def get_vehicle_types():
    """All vehicle types: battery (kWh), consumption (kWh/100 km), charger (kW)."""
    connection = _connect()
    types = pd.read_sql("SELECT * FROM vehicle_types ORDER BY type_id", connection)
    connection.close()
    return types


def get_fleet():
    """One row per van: its data, the name of its type and its km per weekday.

    Columns: vehicle_id, name, type, depot_arrival, depot_departure (as times),
    battery_percent, min_reserve_percent, km_mon ... km_sun
    """
    connection = _connect()
    vans = pd.read_sql("""SELECT f.vehicle_id, f.name, t.name AS type,
                                 f.depot_arrival, f.depot_departure,
                                 f.battery_percent, f.min_reserve_percent
                          FROM vehicles f JOIN vehicle_types t ON t.type_id = f.type_id
                          ORDER BY f.vehicle_id""", connection)
    km = pd.read_sql("SELECT vehicle_id, weekday, km FROM schedule", connection)
    connection.close()

    # The table schedule has one row per van and weekday. pivot() turns it
    # into one row per van with one column per weekday.
    km_wide = km.pivot(index="vehicle_id", columns="weekday", values="km")
    km_wide = km_wide.reindex(columns=range(7), fill_value=0)
    km_wide.columns = KM_COLUMNS
    fleet = vans.merge(km_wide, left_on="vehicle_id", right_index=True, how="left")
    fleet[KM_COLUMNS] = fleet[KM_COLUMNS].fillna(0)

    # "18:00" (text in the database) -> a time, which the table editor needs
    for column in ["depot_arrival", "depot_departure"]:
        fleet[column] = pd.to_datetime(fleet[column], format="%H:%M").dt.time
    return fleet


def save_fleet(fleet, types):
    """Write the edited fleet back into the tables vehicles and schedule.

    The simplest safe way: delete all vans and their km, then insert the
    table again. Vans keep their number (vehicle_id), new vans get a new one.
    Only call this when check_fleet() found no problems.
    """
    type_id_of = dict(zip(types["name"], types["type_id"]))
    connection = _connect()
    cursor = connection.cursor()
    cursor.execute("DELETE FROM schedule")
    cursor.execute("DELETE FROM vehicles")
    for _, van in fleet.iterrows():
        van_id = None if pd.isna(van["vehicle_id"]) else int(van["vehicle_id"])
        cursor.execute("""INSERT INTO vehicles (vehicle_id, name, type_id, depot_arrival,
                          depot_departure, battery_percent, min_reserve_percent)
                          VALUES (?, ?, ?, ?, ?, ?, ?)""",
                       (van_id, van["name"], int(type_id_of[van["type"]]),
                        van["depot_arrival"].strftime("%H:%M"), van["depot_departure"].strftime("%H:%M"),
                        float(van["battery_percent"]), float(van["min_reserve_percent"])))
        van_id = cursor.lastrowid  # the number SQLite gave the van (new or kept)
        for day, column in enumerate(KM_COLUMNS):
            cursor.execute("INSERT INTO schedule VALUES (?, ?, ?)", (van_id, day, float(van[column])))
    connection.commit()
    connection.close()


# ---------------------------------------------------------------------------
# 3. Prices
# ---------------------------------------------------------------------------
def database_ready():
    """True if electricity_prices.db already holds the prices from BuildDB.py."""
    # Check the file first: sqlite3.connect() would create an empty one.
    if not os.path.exists(DATABASE):
        return False
    try:
        connection = sqlite3.connect(DATABASE)
        count = connection.execute("SELECT COUNT(*) FROM prices_ch").fetchone()[0]
        connection.close()
    except sqlite3.Error:
        return False  # the file exists, but BuildDB.py has not run yet
    return count > 0


def _last_known_hour():
    """The last hour with a published price, in Swiss time."""
    connection = sqlite3.connect(DATABASE)
    last = connection.execute("SELECT MAX(time_utc) FROM prices_ch").fetchone()[0]
    connection.close()
    return pd.Timestamp(last, tz="UTC").tz_convert(TZ)


def _prices_from_database(hours):
    """Exchange price in Rp./kWh for each hour, and whether it is a forecast.

    Published hours come from the table prices_ch. For the hours after
    that, the ML forecast from the table predictions (C) is used. While C's
    model has written nothing yet, the price of the same hour one week
    earlier stands in, as BuildDB.py suggests.
    """
    # The database stores UTC text, so we look the hours up the same way
    keys = hours.tz_convert("UTC").strftime(TIME_FORMAT)
    keys_week_before = (hours - pd.Timedelta(days=7)).tz_convert("UTC").strftime(TIME_FORMAT)

    connection = _connect()  # also makes sure the table predictions exists
    known = pd.read_sql("SELECT time_utc, price_rp_kwh FROM prices_ch "
                        "WHERE time_utc >= ? AND time_utc <= ?", connection,
                        params=(min(keys_week_before), max(keys)))
    # ORDER BY created_at: if an hour was forecast twice, the newest comes last
    forecasts = pd.read_sql("SELECT time_utc, predicted_rp_kwh FROM predictions "
                            "WHERE time_utc >= ? AND time_utc <= ? ORDER BY created_at",
                            connection, params=(min(keys), max(keys)))
    connection.close()

    # Dictionaries: hour text -> price (a later value overwrites an earlier one)
    known_price = dict(zip(known["time_utc"], known["price_rp_kwh"]))
    forecast_price = dict(zip(forecasts["time_utc"], forecasts["predicted_rp_kwh"]))

    prices, is_forecast = [], []
    for key, key_week_before in zip(keys, keys_week_before):
        if key in known_price:                   # published by the exchange
            prices.append(known_price[key])
            is_forecast.append(False)
        elif key in forecast_price:              # C's ML forecast
            prices.append(forecast_price[key])
            is_forecast.append(True)
        else:                                    # placeholder: same hour last week
            prices.append(known_price.get(key_week_before, np.nan))
            is_forecast.append(True)
    return prices, is_forecast


# A typical winter day in EUR/MWh, one value per hour (index 0 = 00:00).
# Only used while BuildDB.py has not downloaded the real prices yet.
_WINTER_SHAPE = [96, 88, 80, 76, 78, 90, 118, 140, 134, 118, 104, 96,
                 90, 88, 94, 104, 122, 145, 158, 150, 133, 120, 110, 102]


def _fake_eur_mwh(ts):
    """Make up a believable price for one hour (same input = same output)."""
    rng = np.random.default_rng(ts.date().toordinal())  # one seed per day
    day_factor = 0.85 + 0.3 * rng.random()
    if ts.weekday() >= 5:  # weekends are cheaper
        day_factor *= 0.85
    noise = rng.normal(0, 6, size=24)[ts.hour]
    return round(_WINTER_SHAPE[ts.hour] * day_factor + noise, 1)


def published_until(now=None):
    """Return the first hour whose day-ahead price is NOT published yet."""
    if database_ready():
        return _last_known_hour() + timedelta(hours=1)
    # Without the database: tomorrow's prices appear around 13:00 today
    now = now or datetime.now(TZ)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    days = 2 if now.hour >= 13 else 1
    return midnight + timedelta(days=days)


def get_market_prices(start, end):
    """Exchange price (Rp./kWh) for every hour from `start` to `end` (excluded).

    Returns a DataFrame with ts (Swiss time), market_rp and is_forecast.
    """
    hours = pd.date_range(pd.Timestamp(start).floor("h"), pd.Timestamp(end),
                          freq="h", inclusive="left")
    df = pd.DataFrame({"ts": hours})
    if database_ready():
        df["market_rp"], df["is_forecast"] = _prices_from_database(hours)
    else:
        # EUR/MWh / 10 = euro cents per kWh; x EUR/CHF = Rappen per kWh
        df["market_rp"] = [_fake_eur_mwh(t) / 10 * EUR_CHF for t in df["ts"]]
        df["is_forecast"] = df["ts"] >= pd.Timestamp(published_until())
    # If the database is very old, some hours have no price at all:
    # fill them with the price of the hour before
    df["market_rp"] = df["market_rp"].ffill().bfill()
    return df


def get_prices(tariff_id, start, end):
    """What the company pays per hour: market price + the tariff's surcharge.

    Args:
        tariff_id: a tariff_id from the table tariffs.
        start, end: datetimes in Swiss time (`end` is not included).

    Returns:
        DataFrame with columns ts, eur_mwh, rp_kwh, is_forecast.
    """
    df = get_market_prices(start, end)
    tariffs = get_tariffs()
    tariff = tariffs[tariffs["tariff_id"] == tariff_id].iloc[0]

    # Normal surcharge, or the peak surcharge from peak_start to peak_end (exclusive)
    surcharge = pd.Series(tariff["surcharge_rp_kwh"], index=df.index)
    if not pd.isna(tariff["peak_surcharge_rp_kwh"]):
        hour = df["ts"].dt.hour
        is_peak = (hour >= tariff["peak_start"]) & (hour < tariff["peak_end"])
        surcharge[is_peak] = tariff["peak_surcharge_rp_kwh"]

    df["rp_kwh"] = (df["market_rp"] + surcharge).round(2)
    df["eur_mwh"] = (df["market_rp"] / EUR_CHF * 10).round(1)
    return df[["ts", "eur_mwh", "rp_kwh", "is_forecast"]]


def data_source():
    """One sentence for the sidebar: where the prices on screen come from."""
    if not database_ready():
        return ("No prices in electricity_prices.db yet, so they are made up. "
                "Run `python BuildDB.py` once (a few minutes).")
    last = _last_known_hour()
    text = f"Real prices until {last:%a %d %b %H:00}."
    if last < datetime.now(TZ) - timedelta(days=1):
        text += " The database is out of date: run `python BuildDB.py`."
    return text


def save_charging_plan(blocks_of_van):
    """Store a charging plan in the table charging_plan: one row per van and hour.

    blocks_of_van: dictionary vehicle_id -> list of (ts, kwh) from the planner.
    Old plans stay in the table (created_at tells them apart).
    """
    created = datetime.now(TZ).astimezone(ZoneInfo("UTC")).strftime(TIME_FORMAT)
    rows = []
    for van_id, blocks in blocks_of_van.items():
        for ts, kwh in blocks:
            rows.append((created, int(van_id), ts.tz_convert("UTC").strftime(TIME_FORMAT), kwh))
    connection = _connect()
    connection.executemany("INSERT OR REPLACE INTO charging_plan VALUES (?, ?, ?, ?)", rows)
    connection.commit()
    connection.close()


def get_meter(typical_kwh, days=30):
    """Simulated meter data: kWh charged per van per night, last `days` days.

    typical_kwh: dictionary van name -> kWh it usually charges per night.
    """
    today = datetime.now(TZ).date()
    rows = []
    for i, (name, typical) in enumerate(typical_kwh.items()):
        rng = np.random.default_rng(1000 + i)  # fixed seed per van
        for d in range(days, 0, -1):
            day = today - timedelta(days=d)
            # Saturday night: nothing to charge, no route on Sunday
            kwh = 0.0 if day.weekday() == 5 else max(0.0, rng.normal(typical, typical * 0.15))
            rows.append({"date": pd.Timestamp(day), "name": name, "kwh": round(kwh, 1)})
    return pd.DataFrame(rows)
