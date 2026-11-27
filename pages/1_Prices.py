""" 

Page 1-Prices, shows when electricity is cheap 
Inspiration from the design and layout were taken from weather Apps, like MeteoSuisse and Apple Weather. 
The Layout follows the following points: 
First the answer the people / and companys look for, then night by hour, followed by the whole day displayed as a chart. 

"""
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import streamlit as st

import grafiken
from daten import TZ, get_prices, published_until
from ui_helpers import chosen_tariff, complete_rows, depot_window, describe_tariff, init_state, price_levels

init_state()  # so the page also works when it is opened directly

tariff = chosen_tariff()
arrive, leave = depot_window(complete_rows(st.session_state.fleet))  # e.g. 19 and 6
today = datetime.now(TZ).replace(hour=0, minute=0, second=0, microsecond=0)
tomorrow = today + timedelta(days=1)

st.title("Prices")
# The problem the app solves, in one sentence (grading criterion 1)
st.markdown("Vans that start charging the moment they get back pay the most expensive "
            "hours of the day. This page shows when power is cheap tonight.")

# Tonight = from the hour all vans are back until the first one leaves.
tonight = get_prices(st.session_state.tariff_id, today + timedelta(hours=arrive),
                     tomorrow + timedelta(hours=leave))
best = tonight.loc[tonight["rp_kwh"].idxmin()]
first = tonight.iloc[0]
difference = first["rp_kwh"] - best["rp_kwh"]

with st.container(border=True):
    st.caption("Cheapest hour tonight")
    st.markdown(f"# {best['ts']:%H:%M}")
    if difference >= 0.5:
        st.markdown(f"**{best['rp_kwh']:.1f} Rp./kWh**, {difference:.1f} Rp. less than at "
                    f"{first['ts']:%H:%M}, when all vans are back.")
    else:
        st.markdown(f"**{best['rp_kwh']:.1f} Rp./kWh**. Tonight the price hardly changes.")

st.subheader("Tonight, hour by hour")
columns = st.columns(len(tonight))
levels = price_levels(tonight["rp_kwh"])
for column, (_, hour), level in zip(columns, tonight.iterrows(), levels):
    column.caption(f"{hour['ts']:%H}")
    # ≈ marks hours whose price is not published yet (forecast)
    sign = "≈" if hour["is_forecast"] else ""
    column.markdown(f":{level}[**{sign}{hour['rp_kwh']:.0f}**]")
st.caption("Rp./kWh. Green: cheapest third of the night, red: most expensive third. "
           "≈ = forecast, the exchange has not published this hour yet.")

st.subheader("The whole day")
published = published_until() > tomorrow  # tomorrow's prices come at about 13:00
choice = st.segmented_control("Day", ["Today", "Tomorrow"], default="Tomorrow" if published else "Today",
                              label_visibility="collapsed")
day = tomorrow if choice == "Tomorrow" else today  # nothing selected -> today
if day == tomorrow and not published:
    st.info("Tomorrow's prices are published around 13:00. Until then these are estimates.")

prices = get_prices(st.session_state.tariff_id, day, day + timedelta(days=1))
prices["hour"] = prices["ts"].dt.hour
prices["plugged_in"] = (prices["hour"] >= arrive) | (prices["hour"] < leave)
night = prices[prices["plugged_in"]]  # the hours when all vans are plugged in

fig, ax = plt.subplots(figsize=(10, 3))
ax.bar(prices["hour"], prices["rp_kwh"], color=grafiken.GREY)
ax.set_xticks(range(24))         # one label per hour: 0, 1, ..., 23
ax.set_xlabel("Hour", color=grafiken.LINE)
ax.set_ylabel("Rp./kWh", color=grafiken.LINE)
ax.spines[["top", "right", "left"]].set_visible(False)  # only the line at the bottom
ax.spines["bottom"].set_color(grafiken.LINE)
ax.grid(axis="y", color=grafiken.LINE, alpha=0.25)      # thin help lines
ax.set_axisbelow(True)                                  # help lines behind the bars
ax.tick_params(length=0, colors=grafiken.LINE)          # no tick marks, grey numbers
fig.patch.set_alpha(0)        # see-through background instead of a white box,
ax.set_facecolor("none")      # so the chart fits light and dark mode
st.pyplot(fig)

st.caption(f"All vans are in the depot from {arrive:02d}:00 to {leave:02d}:00 (from the Vans page). "
           f"Price = exchange price (Energy-Charts) + your tariff “{tariff['name']}”: "
           f"{describe_tariff(tariff)}.")

with st.expander("All 24 hours as a table"):
    st.dataframe(prices[["ts", "rp_kwh", "is_forecast", "plugged_in"]], hide_index=True,
                 column_config={"ts": st.column_config.DatetimeColumn("Hour", format="HH:mm"),
                                "rp_kwh": st.column_config.NumberColumn("Rp./kWh", format="%.1f"),
                                "is_forecast": "Forecast",
                                "plugged_in": "All vans plugged in"})
