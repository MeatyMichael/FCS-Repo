""" Depot displayed like a smart home app """

from datetime import timedelta

import streamlit as st

import grafiken
from daten import KM_COLUMNS, get_meter, get_vehicle_types
from depot import changes, soc_at, status_at
from ui_helpers import complete_rows, init_state, with_types

init_state()

# The plans come from the "Simulate tonight" button on the Plan page
plans = st.session_state.plans
if not plans:
    st.title("Depot")
    st.info("Nothing to simulate yet. Open the Plan page and press “Simulate tonight”.")
    st.page_link("pages/3_Plan.py", label="Go to Plan", icon=":material/battery_charging_full:")
    st.stop()


def advance_clock():
    """Button callback: move the simulated clock one hour forward.

    A callback runs BEFORE the page is drawn again, so the title and the
    tiles already show the new time.
    """
    old = st.session_state.clock
    new = old + timedelta(hours=1)
    st.session_state.clock = new
    # e.g. ["Van 1 started charging"], shown as pop-ups below
    st.session_state.pending_toasts = changes(plans, old, new)


def reset_clock():
    """Button callback: back to the moment the vans were plugged in."""
    st.session_state.clock = st.session_state.sim_start
    st.session_state.pending_toasts = []


# Notifications from the last "+1 hour" click
for message in st.session_state.pending_toasts:
    st.toast(message, icon=":material/ev_station:")
st.session_state.pending_toasts = []

# The simulated time and the buttons to move it forward. The title is updated on every click.
clock = st.session_state.clock
st.title(f"Depot · {clock:%a %H:%M}")
st.caption("Simulated time. Press “+1 hour” to let the night go on.")
# Buttons on the left: pop-ups appear on the right and would cover them there.
b1, b2, _ = st.columns([1, 1, 4])
b1.button("+1 hour", on_click=advance_clock, type="primary", width="stretch")
b2.button("Reset", on_click=reset_clock, width="stretch")

# One tile per vehicle
# status_at() says for each van: "scheduled", "charging" or "ready",
# e.g. {"Van 1": "charging", "Van 2": "ready"}
statuses = status_at(plans, clock)

# What each status looks like: text and badge colour
LOOK = {
    "scheduled": ("Starts at {start:%H:%M}", "blue"),
    "charging": ("Charging until {end:%H:%M}", "green"),   # green like a charging phone
    "ready": ("Ready", "gray"),
}

# One tile per vehicle, two per row. Like the battery widget on a phone:
# the name and what it is doing on the left, the charge level big on the right.
columns = st.columns(2)
for i, plan in enumerate(plans):
    with columns[i % 2].container(border=True):
        status = statuses.get(plan["name"])
        text, colour = LOOK.get(status, ("?", "gray"))
        soc = soc_at(plan, clock)
        left, right = st.columns([3, 1], vertical_alignment="center")
        left.markdown(f"**{plan['name']}**")
        left.badge(text.format(**plan), color=colour)
        right.markdown(f"## {soc:.0f} %", text_alignment="right", anchors=False)
        st.progress(min(1.0, soc / 100))

#The whole night as a timeline 
st.subheader("Tonight's plan")
st.altair_chart(grafiken.timeline_chart(plans, clock), width="stretch")
st.caption("Green: when each vehicle charges. Blue line: the simulated time.")

#Simulated electricity meter (Salman) asked for simulated consumption data)
st.subheader("Depot consumption, last 30 days")
# A van charges about what its average route uses (Monday to Saturday)
vans = complete_rows(with_types(st.session_state.fleet, get_vehicle_types()))
average_kwh = vans[KM_COLUMNS[:6]].mean(axis=1) * vans["consumption_kwh_100km"] / 100
typical_kwh = dict(zip(vans["name"], average_kwh))
meter = get_meter({p["name"]: typical_kwh.get(p["name"], 20) for p in plans})
st.altair_chart(grafiken.consumption_chart(meter), width="stretch")
st.caption("Simulated meter data (fixed random seed), based on each vehicle's average route.")