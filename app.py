"""

app.py is the entry point of the app: page setup, menu and sidebar as well as other features are determined here. 

"""
from pathlib import Path

import streamlit as st

import BuildDB
from daten import data_source, database_ready, get_tariffs, save_setting
from ui_helpers import chosen_tariff, company, describe_tariff, init_state
# 
st.set_page_config(page_title="VOLT", page_icon=":material/ev_station:", layout="wide")

# this is supposed to add our own simple Logo (images/), shown at the top of the sidebar. 
# icon_image is the small version for when the sidebar is closed. 
IMAGES = Path(__file__).parent / "images" # the folder images/ 
st.logo(IMAGES / "logo.png", icon_image=IMAGES / "icon.png", size="large")
# The menu: Each page is a file in pages/. 
# A dictionary makes groups: "" has no heading, "Settings" gets a small heading.
pages = {
    "": [
        st.Page("pages/1_Prices.py", title="Prices", icon=":material/show_chart:", default=True),
        st.Page("pages/2_Fleet.py", title="Fleet", icon=":material/local_shipping:"),
        st.Page("pages/3_Plan.py", title="Plan", icon=":material/battery_charging_full:"),
        st.Page("pages/4_Depot.py", title="Depot", icon=":material/ev_station:"),
        st.Page("pages/6_Forecast.py", title="Forecast", icon=":material/insights:"),
        st.Page("pages/5_Savings.py", title="Savings (bonus)", icon=":material/savings:"),
    ],
    "Settings": [
        st.Page("pages/7_Company.py", title="Company", icon=":material/storefront:"),
    ],
}
navigation = st.navigation(pages)

# Keep the prices up to date with Michael's function from BuildDB.py. It only downloads the missing hours 
# (a few seconds) and tries at most every 30 minutes. The very first download (a few minutes) is done by hand with
# `python BuildDB.py`, so the app never hangs on the first start.
if database_ready():
    with st.spinner("checking for new electricity prices..."):
        notice = BuildDB.update_if_needed()
    if notice:
        st.sidebar.info(notice)

init_state()

def remember_tariff():
    """Callback of the tariff box: store choice in electricity_prices.db for next time."""
    save_setting("selected_tariff", st.session_state.tariff_id)

tariffs = get_tariffs()
tariff_names = dict(zip(tariffs["tariff_id"], tariffs["name"]))  # 1 -> "Exchange price only", ...
# this defines the sidebar so it is the same on every page. 
# Widgets with a 'key' write their value into st.session_state[key]

profile = company()
with st.sidebar: 
    # Defines who is using the app, (the company's account)
    if profile["company_name"]:
        st.markdown(f"**{profile['company_name']}**")
        depot = profile["depot_address"] or "Depot"
        st.caption(f"{depot} · {st.session_state.chargers} chargers")
    else:
        st.page_link("pages/7_Company.py", label="Set up your company", icon=":material/add_business:")
    st.selectbox("Tariff", [int(i) for i in tariff_names], key="tariff_id",
                 format_func=lambda i: tariff_names[i], on_change=remember_tariff,
                 help="From the table tariffs: what your utility adds to the exchange price.")
    st.caption(describe_tariff(chosen_tariff()))
    st.divider()
    if database_ready():
        st.caption(data_source())   # "Real prices until Fri 27 Nov 23:00."
    else:
        st.warning(data_source())   # no prices yet: say so clearly
    st.caption("Prices: Energy-Charts (CC BY 4.0). Weather: Open-Meteo.com (CC BY 4.0).")

navigation.run()
