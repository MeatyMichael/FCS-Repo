"""
Page 7 · Company - the company's profile and depot (role B, Cosmo).


Why there is no login: each company runs its own copy of the app with its
own electricity_prices.db, so nobody else sees its vans. The profile is
saved in the table settings (name, value), like the chosen tariff.
"""

import streamlit as st

from daten import save_setting
from ui_helpers import company, init_state

init_state()
profile = company()   # what is saved so far, e.g. profile["company_name"] ("" if empty)

st.title("Company")
st.markdown("Who uses the app and where the vans charge. "
            "It is saved in electricity_prices.db, so it is still there next time.")

# A message from the last click on Save (shown once, after the page reloads)
if "company_message" in st.session_state:
    st.success(st.session_state.pop("company_message"))

# st.form: the page only reacts when "Save changes" is pressed, not after
# every letter. All values come back at once.
with st.form("company"):
    # Each field starts with what is saved (value=...), so the user can change it.
    name = st.text_input("Company name", value=profile["company_name"], required=True,
                         placeholder="e.g. Muster Transport AG")
    address = st.text_input("Depot address", value=profile["depot_address"],
                            placeholder="e.g. Industriestrasse 1, 9000 St. Gallen")
    chargers = st.number_input("Chargers in the depot", min_value=1, max_value=200,
                               value=st.session_state.chargers,
                               help="Used on the Plan page: a warning appears if there are more vans than chargers.")
    # Three fields side by side
    left, middle, right = st.columns(3)
    contact = left.text_input("Contact person", value=profile["contact_name"], placeholder="e.g. Anna Muster")
    # type="email" and type="phone" give the right keyboard on a phone and a
    # small icon; "email" also checks the format in the browser.
    email = middle.text_input("Email", value=profile["contact_email"], type="email",
                               placeholder="e.g. fleet@muster.ch")
    phone = right.text_input("Phone", value=profile["contact_phone"], type="phone",
                              placeholder="e.g. 071 123 45 67")
    saved = st.form_submit_button("Save changes", type="primary")  # True right after the click

if saved:
    # The browser already checks these, but a check in Python cannot be skipped.
    if name.strip() == "":
        st.error("Please enter the company name.")
    elif email.strip() != "" and "@" not in email:
        st.error(f"“{email}” is not an email address.")
    else:
        # Save every field in the table settings; .strip() removes spaces at the ends
        save_setting("company_name", name.strip())
        save_setting("depot_address", address.strip())
        save_setting("contact_name", contact.strip())
        save_setting("contact_email", email.strip())
        save_setting("contact_phone", phone.strip())
        save_setting("num_charging_stations", chargers)
        st.session_state.chargers = int(chargers)   # the Plan page reads it from here
        st.session_state.company_message = "Saved. The sidebar now shows your company."
        st.rerun()  # draw the page again, so the sidebar shows the new name

st.subheader("Light or dark")
st.markdown("The app follows your computer, like the apps on a Mac or iPhone: light during "
            "the day, dark at night if your computer switches. To choose yourself, open the "
            "**⋮** menu at the top right and pick **Light** or **Dark**.")

with st.expander("Why is there no login?"):
    st.markdown("Each company runs its own copy of the app with its own database file, so "
                "nobody else can see its vans and nobody needs a password. A login would only "
                "be needed if many companies shared one app on the internet. Then every table "
                "would also need a column that says which company a row belongs to.")