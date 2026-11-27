""" Page 3 · Plan - when each van charges tonight """

from datetime import timedelta

import pandas as pd
import streamlit as st

import grafiken
from checks import check_fleet
from daten import get_prices, get_vehicle_types, save_charging_plan
from depot import make_plan
from planer import cost_chf, cost_on_arrival_chf, plan_ev_charging
from ui_helpers import (complete_rows, fmt_hours, init_state, next_departure,
                        plug_in_tonight, route_kwh, with_types)

init_state()
types = get_vehicle_types()

st.title("Plan")

# Stop if there is no information to plan

fleet = st.session_state.fleet.dropna(subset=["name"])
if fleet.empty:
    st.info("No vehicles yet. Add them on the Fleet page.")
    st.page_link("pages/2_Fleet.py", label="Go to Vehicles", icon=":material/local_shipping:")
    st.stop()

try:
    errors = check_fleet(fleet, types)
except NotImplementedError:
    errors = []  # Exercise 2 not done yet: plan without checks
if errors:
    st.error("Please fix the table of vehicles first:\n\n- " + "\n- ".join(errors))
    st.page_link("pages/2_Fleet.py", label="Go to Vehicles", icon=":material/local_shipping:")
    st.stop()

# Battery consumption and charger power depend on the values of its dedicated type

vehicles = complete_rows(with_types(fleet, types))
if len(vehicles) > st.session_state.chargers:
    st.warning(f"{len(vehicles)} vans but only {st.session_state.chargers} chargers. "
               "The plan assumes every van has its own charger. "
               "You can change the number of chargers on the Company page.")

# The two nights the planner may use: tonight (real prices) and the next
# night (partly forecast). Prices from the first arrival tonight until the
# last departure the morning after next.
first_plug_in = plug_in_tonight(min(vehicles["depot_arrival"]))
last_departure = max(next_departure(row["depot_departure"], plug_in_tonight(row["depot_arrival"]))
                     for _, row in vehicles.iterrows())
prices = get_prices(st.session_state.tariff_id, first_plug_in, last_departure + timedelta(days=1))

# Plan each van on its own. The lists and dictionaries collect the results:
# rows -> the table "All numbers", cards -> one row per van, load -> the fleet
# chart, plans -> the Depot page, blocks_of -> "One van in detail",
# plan_of_van_id -> the table charging_plan in the database.
rows, cards, load, plans, blocks_of, plan_of_van_id = [], [], [], [], {}, {}
for _, van in vehicles.iterrows():
    # The two nights of this van: plugged in tonight until tomorrow morning,
    # and the same times one day later.
    plug_in_1 = plug_in_tonight(van["depot_arrival"])
    dep_1 = next_departure(van["depot_departure"], plug_in_1)
    plug_in_2 = plug_in_1 + timedelta(days=1)
    dep_2 = dep_1 + timedelta(days=1)
    # Only full hours while the van is plugged in (an hour must end before departure).
    ts = prices["ts"]
    plugged = ((ts >= plug_in_1) & (ts + timedelta(hours=1) <= dep_1)) | \
              ((ts >= plug_in_2) & (ts + timedelta(hours=1) <= dep_2))
    van_prices = prices[plugged]

# How full must the battery be? Reserve + the km of the next routes.
    battery = van["battery_kwh"]
    need_1 = route_kwh(van, dep_1)   # tomorrow's route (kWh)
    need_2 = route_kwh(van, dep_2)   # the route the day after
    reserve = van["min_reserve_percent"] / 100
    soc_min = min(1.0, reserve + need_1 / battery)             # must be reached by dep_1
    soc_target = min(1.0, reserve + (need_1 + need_2) / battery)  # charged when it is cheap

    # C's planner picks the cheapest hours. blocks = [(hour, kWh), ...]
    try:
        blocks = plan_ev_charging(battery, van["battery_percent"] / 100, soc_min,
                                  dep_1, soc_target, van["charging_power_kw"], van_prices)
    except ValueError as err:
        st.error(f"{van['name']}: {err}")
        continue

    # Remember the result for the detail view, the database and the Depot page
    blocks_of[van["name"]] = (blocks, van, dep_1)
    if not pd.isna(van["vehicle_id"]):  # vans not saved yet have no number
        plan_of_van_id[van["vehicle_id"]] = blocks
    plan = make_plan(van["name"], blocks, battery, van["battery_percent"], van["charging_power_kw"],
                     plug_in_1, dep_1)
    plans.append(plan)

    # Compare: the plan's cost against charging right away when the van arrives
    kwh_tonight = sum(k for t, k in blocks if t < dep_1)
    kwh_next = sum(k for t, k in blocks if t >= dep_1)
    total_kwh = kwh_tonight + kwh_next
    plan_cost = cost_chf(blocks, van_prices)
    arrival_cost = cost_on_arrival_chf(total_kwh, van["charging_power_kw"], van_prices)
    cards.append({"name": van["name"], "type": van["type"], "plan": plan, "departure": dep_1,
                  "saved": arrival_cost - plan_cost})
    # One line for the table "All numbers"
    rows.append({
        "Van": van["name"],
        "Tomorrow's route (kWh)": need_1,
        "Charging tonight": fmt_hours(plan["start"], plan["end"]),
        "kWh tonight": kwh_tonight,
        "% at departure": plan["soc_end"],
        "Rest next night (kWh)": kwh_next,
        "Plan (CHF)": plan_cost,
        "On arrival (CHF)": arrival_cost,
        "Saved (CHF)": round(arrival_cost - plan_cost, 2),
    })
    # One bar piece per charging hour for the fleet chart
    for t, k in blocks:
        load.append({"ts": t, "ts_end": t + timedelta(hours=1), "name": van["name"], "kwh": k})

if not rows:  # no van could be planned (the errors are shown above)
    st.stop()

table = pd.DataFrame(rows)

# Show answer in a big numnber 
with st.container(border=True):
    st.caption(f"Tonight ({first_plug_in:%a %d %b}) and the night after, "
               "compared with charging as soon as the vans arrive")
    st.markdown(f"# Saves CHF {table['Saved (CHF)'].sum():.2f}")
    st.markdown(f"**{len(rows)} of {len(vehicles)} vans** are ready on time. "
                f"Tonight the fleet charges {table['kWh tonight'].sum():.0f} kWh.")

# Displayment of the Fleet 
for card in cards:
    plan = card["plan"]
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2, 2, 3, 1], vertical_alignment="center")
        c1.markdown(f"**{card['name']}**")
        c1.caption(card["type"])
        hours = fmt_hours(plan["start"], plan["end"])
        c2.markdown(f"Charges {hours}" if hours != "–" else "No charging needed tonight")
        c3.progress(min(1.0, plan["soc_end"] / 100),
                    text=f"{plan['soc_end']} % when it leaves at {card['departure']:%H:%M}")
        c4.markdown(f"**CHF {card['saved']:.2f}**")
        c4.caption("saved")

# Fleet per hour 
st.subheader("Hour by hour")
# Show the price line only for the two nights (not for the day in between).
one_day = timedelta(days=1)
night_hours = ((prices["ts"] >= first_plug_in) & (prices["ts"] < last_departure)) | \
              ((prices["ts"] >= first_plug_in + one_day) & (prices["ts"] < last_departure + one_day))
load = pd.DataFrame(load, columns=["ts", "ts_end", "name", "kwh"])  # columns also when empty
st.altair_chart(grafiken.fleet_load_chart(load, prices[night_hours]), width="stretch")
st.caption("Bars: kWh per van and hour. Line: price in Rp./kWh, dashed = forecast.")

# Details, hidden but clickable 
with st.expander("All numbers"):
    # Number formats: 1 decimal for kWh, 2 for CHF. The | joins the three dictionaries.
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.1f") for c in
                                ["Tomorrow's route (kWh)", "kWh tonight", "Rest next night (kWh)"]}
                 | {c: st.column_config.NumberColumn(format="%.2f") for c in
                    ["Plan (CHF)", "On arrival (CHF)", "Saved (CHF)"]}
                 | {"% at departure": st.column_config.NumberColumn(format="%d %%")})
    st.caption("Next-night prices are partly a forecast.")

with st.expander("One van in detail"):
    chosen = st.selectbox("Van", list(blocks_of))
    blocks, van, dep_1 = blocks_of[chosen]
    st.altair_chart(grafiken.charge_curve(blocks, van["battery_kwh"], van["battery_percent"], dep_1),
                    width="stretch")

# Make the plan visible on the depot page 
if st.button("Simulate tonight", type="primary"):
    st.session_state.plans = plans
    st.session_state.sim_start = first_plug_in  # the simulation starts when the first van arrives
    st.session_state.clock = first_plug_in
    st.session_state.pending_toasts = []
    save_charging_plan(plan_of_van_id)          # also kept in electricity_prices.db (table charging_plan)
    st.success("Plan saved. Watch it run on the Depot page.")
    st.page_link("pages/4_Depot.py", label="Go to Depot", icon=":material/ev_station:")
