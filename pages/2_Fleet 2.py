"""Page 2 · the fleet, editable like a spreadsheet """

from datetime import datetime, time, timedelta

import streamlit as st

from checks import check_fleet
from daten import KM_COLUMNS, TZ, get_fleet, get_vehicle_types, save_fleet
from ui_helpers import complete_rows, init_state, route_kwh, with_types

init_state()
types = get_vehicle_types()

st.title("Fleet")
st.markdown("Your vehicles and their usual week. Each evening, update how full each battery is. "
            "Changes count right away; **Save changes** keeps them for next time.")

# A message from the last click on Save (shown once, after the page reloads)
if "fleet_message" in st.session_state:
    st.success(st.session_state.pop("fleet_message"))

if "fleet_editor" not in st.session_state:
    st.session_state.fleet_base = st.session_state.fleet.copy()

columns = {
    "vehicle_id": None,  # None hides the column: the database needs the number, the user does not
    "name": st.column_config.TextColumn("Van", required=True),
    "type": st.column_config.SelectboxColumn("Type", options=list(types["name"]), required=True),
    "depot_arrival": st.column_config.TimeColumn("Arrives", format="HH:mm", default=time(18, 0)),
    "depot_departure": st.column_config.TimeColumn("Leaves", format="HH:mm", default=time(7, 0)),
    "battery_percent": st.column_config.NumberColumn(
        "Charge on arrival (%)", min_value=0, max_value=100, step=1,
        help="How full the battery is when the van comes back. Update it every evening."),
    "min_reserve_percent": st.column_config.NumberColumn(
        "Reserve (%)", min_value=0, max_value=100, step=1, default=30,
        help="The battery never goes below this, for detours and cold days."),
}
for column, day in zip(KM_COLUMNS, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
    columns[column] = st.column_config.NumberColumn(f"{day} km", min_value=0, step=1)

edited = st.data_editor(
    st.session_state.fleet_base,
    key="fleet_editor",
    num_rows="dynamic",          # lets the user add and delete vans
    hide_index=True,
    width="stretch",
    column_config=columns,
)
st.session_state.fleet = edited

try:
    errors = check_fleet(edited, types)
except NotImplementedError:
    errors = None
    st.warning("Exercise 2 is still open: write check_fleet() in checks.py. Until then nothing is checked.")

if errors:
    for message in errors:
        st.error(message)
elif errors == []:
    st.success("All vehicles look fine.")


def reload_from_database():
    """Show the fleet exactly as it is stored in electricity_prices.db."""
    st.session_state.fleet = get_fleet()
    st.session_state.fleet_base = st.session_state.fleet.copy()
    del st.session_state["fleet_editor"]  # the editor starts again from fleet_base
    st.rerun()

# Summary
# --- Summary in one sentence -------------------------------------------------
tomorrow = datetime.now(TZ).date() + timedelta(days=1)
vans = complete_rows(with_types(edited, types))
st.markdown(f"**{len(vans)} vans.** {tomorrow:%A}'s routes need "
            f"**{route_kwh(vans, tomorrow).sum():.0f} kWh** (km × what the vehicle type uses).")
# Save to the database
complete = complete_rows(edited)
b1, b2, _ = st.columns([1, 1, 3])
# Saving is only allowed when the checks found nothing (errors is [] or None)
if b1.button("Save changes", type="primary", disabled=bool(errors), width="stretch"):
    save_fleet(complete, types)
    st.session_state.fleet_message = f"Saved {len(complete)} vehicles."
    reload_from_database()
if b2.button("Undo changes", width="stretch"):
    reload_from_database()
if len(complete) < len(edited):
    st.caption("Rows with empty cells are not saved and not planned.")

#Permanent changes to the settings
with st.expander("Vehicle types"):
    st.dataframe(types.drop(columns="type_id"), hide_index=True, width="stretch",
                 column_config={"name": "Type", "battery_kwh": "Battery (kWh)",
                                "consumption_kwh_100km": "Uses (kWh per 100 km)",
                                "charging_power_kw": "Charger (kW)"})
    st.caption("Guide values from BuildDB.py. E adds datasheet sources.")
