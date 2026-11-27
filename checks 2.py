"""
checks.py - input checks for the fleet table (role B, Cosmo).

Generated with Claude (Anthropic) on 19 Nov 2026 (rules and tests),
changed on 23 Nov 2026 for the team database's fleet table; the function
body was written by Claude on 27 Nov 2026. Test: `pytest tests/test_checks.py`.
"""

import pandas as pd  # pd.isna(value) is True for an empty cell

from daten import KM_COLUMNS  # ["km_mon", "km_tue", ..., "km_sun"]


def check_fleet(fleet, types):
    """Check the fleet table and return one error message per problem.

    An empty list means everything is fine. The page shows each message with
    st.error(...), and the fleet is only saved and planned when the list is empty.

    fleet: one row per van with the columns name, type, depot_arrival,
        depot_departure, battery_percent, min_reserve_percent and the
        seven km columns in KM_COLUMNS (percentages are numbers from 0 to 100).
    types: the vehicle types with the columns name, battery_kwh,
        consumption_kwh_100km, charging_power_kw.

    Rules (one message per broken rule and van):
        1. Every van needs a name, and no two vans may have the same name.
        2. type must be one of the names in types.
        3. battery_percent and min_reserve_percent must be between 0 and 100.
        4. depot_arrival and depot_departure must be filled in.
        5. All seven km values must be filled in and not negative.
        6. The longest route must fit into the battery above the reserve:
               longest km x consumption_kwh_100km / 100
               must not be more than  battery_kwh x (100 - reserve) / 100
           Check rule 6 only if rules 2, 3 and 5 are fine for this van.
        (Empty cells in pandas are NaN or None: use pd.isna(value) to test.)

    Every message must contain the van's name, e.g.
        "Transporter 1: 250 km need 70.0 kWh, but only 56.0 kWh are usable"
    A van without a name is named by its row instead: "Row 3: ...".

    Example:
        A fleet where only Tricycle 1 has battery_percent 120 returns
        exactly one message, and "Tricycle 1" is in it.
    """
    errors = []
    type_names = list(types["name"])     # e.g. ["Small electric van", ...]
    type_of = types.set_index("name")    # look up a type by its name:
    #   type_of.loc["Electric tricycle", "battery_kwh"] is 7.0
    all_names = list(fleet["name"])      # to find names that appear twice

    row = 0
    for _, van in fleet.iterrows():      # one van (one row of the table) at a time
        row = row + 1                    # 1, 2, 3 ... for vans without a name
        ok = True                        # False if rule 2, 3 or 5 breaks (rule 6 needs them)

        # Rule 1: a name, and no name twice
        name = van["name"]
        if pd.isna(name) or str(name).strip() == "":
            errors.append(f"Row {row}: the van needs a name")
            name = f"Row {row}"          # so the next messages still say which van
        elif all_names.count(name) > 1:
            errors.append(f"{name}: two vans have this name")

        # Rule 2: a known vehicle type
        if pd.isna(van["type"]) or van["type"] not in type_names:
            errors.append(f"{name}: unknown type ({van['type']})")
            ok = False

        # Rule 3: charge and reserve between 0 and 100 %
        charge = van["battery_percent"]
        reserve = van["min_reserve_percent"]
        if pd.isna(charge) or pd.isna(reserve) or not (0 <= charge <= 100 and 0 <= reserve <= 100):
            errors.append(f"{name}: charge and reserve must be between 0 and 100 %")
            ok = False

        # Rule 4: arrival and departure filled in
        if pd.isna(van["depot_arrival"]) or pd.isna(van["depot_departure"]):
            errors.append(f"{name}: arrival and departure must be filled in")

        # Rule 5: km for all seven weekdays, none negative
        km = van[KM_COLUMNS]             # the seven km values of this van
        if km.isna().any() or (km < 0).any():
            errors.append(f"{name}: every weekday needs km (0 or more)")
            ok = False

        # Rule 6: the longest route must fit into the battery above the reserve
        if ok:
            longest = km.max()
            need = longest * type_of.loc[van["type"], "consumption_kwh_100km"] / 100
            usable = type_of.loc[van["type"], "battery_kwh"] * (100 - reserve) / 100
            if need > usable:
                errors.append(f"{name}: {longest:.0f} km need {need:.1f} kWh, "
                              f"but only {usable:.1f} kWh are usable")
    return errors