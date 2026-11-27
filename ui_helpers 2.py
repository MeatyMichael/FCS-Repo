"""
this file helps define code used by all pages of the app (mainly for UI (user interface) and UE (user experience))
this file continues to be improved and adapted depending on the needs of the team
"""

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from daten import KM_COLUMNS, TZ, get_fleet, get_setting, get_tariffs

def init_state():
    """Puts the default values into st.session_state, but only once.
    Streamlit runs the whole script again on every click, so normal variables start fresh each time. st.session_state works like a dictionary
    and is shared by all pages. 
    The fleet, the tariffs, and the number of chargers come from electricity_prices.db
    """

    defaults = {
        "plans": [],                  # filled by "Simulate tonight" on the Plan page
        "clock": None,                # simulated time on the Depot page
        "sim_start": None,            # when the simulation starts (first van plugged in)
        "pending_toasts": [],         # notifications to show after "+1 hour"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "tariff_id" not in st.session_state:
        ids = [int(i) for i in get_tariffs()["tariff_id"]]
        saved = int(get_setting("selected_tariff", ids[0]))
        st.session_state.tariff_id = saved if saved in ids else ids[0]
    if "chargers" not in st.session_state:
        st.session_state.chargers = int(get_setting("num_charging_stations", 4))
    if "fleet" not in st.session_state:
        st.session_state.fleet = get_fleet()

# The company profile (Company page). Stored in the table settings, like the tariff.
COMPANY_FIELDS = ["company_name", "depot_address", "contact_name", "contact_email", "contact_phone"]


def company():
    """The company profile as a dictionary, e.g. {"company_name": "Post Filiale St. Gallen", ...}.

    A field that was never filled in is an empty string "".
    """
    profile = {}
    for field in COMPANY_FIELDS:
        profile[field] = get_setting(field, "")
    return profile


def chosen_tariff():
    """The row of the table tariffs that is chosen in the sidebar."""
    tariffs = get_tariffs()
    return tariffs[tariffs["tariff_id"] == st.session_state.tariff_id].iloc[0]


def describe_tariff(tariff):
    """The tariff in one short sentence, e.g. for the sidebar."""
    text = f"+{tariff['surcharge_rp_kwh']:.1f} Rp./kWh on the exchange price"
    if not pd.isna(tariff["peak_surcharge_rp_kwh"]):
        text += (f", +{tariff['peak_surcharge_rp_kwh']:.1f} from "
                 f"{int(tariff['peak_start'])}:00 to {int(tariff['peak_end'])}:00")
    if tariff["source"] == "PLACEHOLDER":
        text += " (placeholder values)"
    return text


def price_levels(prices):
    """'green', 'orange' or 'red' for each price: cheapest, middle or most expensive third.

    Like the traffic lights on a smart meter: the colours are relative to
    the hours shown, not fixed limits.
    """
    low = prices.quantile(1 / 3)
    high = prices.quantile(2 / 3)
    levels = []
    for price in prices:
        if price <= low:
            levels.append("green")
        elif price >= high:
            levels.append("red")
        else:
            levels.append("orange")
    return levels


def depot_window(fleet):
    """The full hours when ALL vans are in the depot: (arrive, leave), e.g. (19, 6).

    From the last arrival until the first departure. Times can be compared
    like numbers, so max() and min() work on them.
    """
    if fleet.empty:
        return 18, 7
    last_arrival = max(fleet["depot_arrival"])
    first_departure = min(fleet["depot_departure"])
    # 18:30 -> the first full hour is 19:00
    arrive = last_arrival.hour + 1 if last_arrival.minute > 0 else last_arrival.hour
    return min(arrive, 23), first_departure.hour  # a van back at 23:30 still counts from 23:00


def complete_rows(fleet):
    """Only the vans where every cell is filled in (new rows often are not yet)."""
    columns = [c for c in fleet.columns if c != "vehicle_id"]
    return fleet.dropna(subset=columns)


def with_types(fleet, types):
    """Add the battery, consumption and charger power of each van's type.

    merge() works like a lookup: the column type of the fleet is matched with
    the name of the vehicle type.
    """
    type_data = types.rename(columns={"name": "type"})
    type_data = type_data[["type", "battery_kwh", "consumption_kwh_100km", "charging_power_kw"]]
    return fleet.merge(type_data, on="type", how="left")


def route_kwh(van, day):
    """kWh a van needs for its route on `day`: km of that weekday x consumption.

    Works for one van (a row) and for the whole table at once.
    """
    km = van[KM_COLUMNS[day.weekday()]]
    return km * van["consumption_kwh_100km"] / 100


def plug_in_tonight(arrival):
    """Today's date at the arrival time, e.g. today 18:00 (Swiss time)."""
    today = datetime.now(TZ).date()
    return datetime.combine(today, arrival, tzinfo=TZ)


def next_departure(departure, after):
    """The first time `departure` happens after `after` (normally tomorrow morning)."""
    candidate = datetime.combine(after.date(), departure, tzinfo=TZ)
    if candidate <= after:
        candidate += timedelta(days=1)
    return candidate


def fmt_hours(start, end):
    """'02:00–05:00', or '–' if there is nothing to show."""
    if start is None or end is None or start == end:
        return "–"
    return f"{start:%H:%M}–{end:%H:%M}"
