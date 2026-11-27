"""
grafiken.py - STUB (placeholder) for role D (Charts).

Generated with Claude (Anthropic) on 19 Nov 2026 so that Cosmo's pages show
something before D's real charts exist (colours for light and dark mode
changed on 27 Nov 2026). D replaces or improves these
functions and keeps the names. Every function takes DataFrames and returns
an Altair chart; the page shows it with st.altair_chart(chart).

The price chart on the Prices page is NOT here: Cosmo builds it.
"""

import altair as alt
import pandas as pd

# Apple's system colours, the same as in .streamlit/config.toml (new look, 27 Nov 2026).
# They are picked so that they can be seen on a white page (light mode) and on
# a black page (dark mode), because Python does not know which mode the viewer uses.
GREEN = "#34C759"   # cheap / charging
BLUE = "#0A84FF"    # scheduled / buttons / "now"
GREY = "#AEAEB2"    # normal hours / ready / forecast
LINE = "#8E8E93"    # lines and text in charts: mid grey, readable on white and on black
VAN_COLOURS = [BLUE, GREEN, "#FF9500", "#AF52DE", "#5AC8FA", "#8E8E93"]


def _local(values):
    """Swiss time without the time-zone label.

    The browser draws times in the viewer's own time zone. Without the
    label, 02:00 stays 02:00 on every computer (and on the screenshots).
    """
    values = pd.to_datetime(pd.Series(values))
    if values.dt.tz is not None:
        values = values.dt.tz_convert("Europe/Zurich").dt.tz_localize(None)
    return values


def _vega_time(ts):
    """A time for Altair's axis settings (start and end of an axis)."""
    return alt.DateTime(year=ts.year, month=ts.month, date=ts.day, hours=ts.hour, minutes=ts.minute)


def fleet_load_chart(load, prices):
    """Stacked bars: kWh per hour per van, with the price as a line on top.

    load: columns ts, ts_end, name, kwh
    prices: columns ts, rp_kwh, is_forecast (only the plugged-in hours)
    """
    load = load.assign(ts=_local(load["ts"]).values, ts_end=_local(load["ts_end"]).values)
    prices = prices.assign(
        ts=_local(prices["ts"]).values,
        # One line per night, so the line does not jump across the day
        night=(_local(prices["ts"]) - pd.Timedelta(hours=12)).dt.strftime("%a").values,
        price=prices["is_forecast"].map({False: "real price", True: "forecast"}).values,
    )
    bars = alt.Chart(load).mark_bar(opacity=0.9).encode(
        x=alt.X("ts:T", title=None, axis=alt.Axis(format="%a %H:%M")),
        x2="ts_end:T",
        y=alt.Y("sum(kwh):Q", title="kWh per hour", stack=True),
        color=alt.Color("name:N", title="Van", scale=alt.Scale(range=VAN_COLOURS)),
        tooltip=[alt.Tooltip("name:N", title="Van"),
                 alt.Tooltip("ts:T", title="Hour", format="%a %H:%M"),
                 alt.Tooltip("kwh:Q", title="kWh", format=".1f")],
    )
    line = alt.Chart(prices).mark_line(color=LINE, point=True).encode(
        x="ts:T",
        y=alt.Y("rp_kwh:Q", title="Rp./kWh"),
        detail="night:N",
        strokeDash=alt.StrokeDash("price:N", title="Price", scale=alt.Scale(
            domain=["real price", "forecast"], range=[[1, 0], [5, 4]])),
        tooltip=[alt.Tooltip("ts:T", title="Hour", format="%a %H:%M"),
                 alt.Tooltip("rp_kwh:Q", title="Rp./kWh", format=".1f"),
                 alt.Tooltip("price:N", title="Price")],
    )
    return alt.layer(bars, line).resolve_scale(y="independent").properties(height=320)


def charge_curve(blocks, battery_kwh, soc_now,
                 departure):
    """Charge level of one van over time, one point after each charging hour.

    blocks: list of (ts, kwh) from plan_ev_charging
    soc_now: charge level now in percent
    departure: hours after this belong to the next night (forecast)
    """
    rows, soc = [], soc_now
    for ts, kwh in blocks:
        soc += kwh / battery_kwh * 100
        rows.append({"ts": ts + pd.Timedelta(hours=1), "soc": round(soc, 1),
                     "night": "tonight" if ts < pd.Timestamp(departure) else "next night (forecast)"})
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame({"ts": [pd.Timestamp(departure)], "soc": [soc_now], "night": ["tonight"]})
    df["ts"] = _local(df["ts"]).values
    return alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("ts:T", title=None, axis=alt.Axis(format="%a %H:%M")),
        y=alt.Y("soc:Q", title="Charge level (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("night:N", title=None, scale=alt.Scale(range=[BLUE, GREY])),
    ).properties(height=240)


def timeline_chart(plans, now):
    """One bar per van from start to end of charging, a line at `now`."""
    df = pd.DataFrame([{"name": p["name"], "start": p["start"], "end": p["end"]} for p in plans])
    df["start"], df["end"] = _local(df["start"]).values, _local(df["end"]).values
    # Show one extra hour after the last charging, so the last bar is not cut off
    last = max(df["end"].max(), _local([now])[0]) + pd.Timedelta(hours=1)
    first = df["start"].min()
    domain = [_vega_time(first), _vega_time(last)]
    bars = alt.Chart(df).mark_bar(color=GREEN, cornerRadius=9, height=18).encode(
        x=alt.X("start:T", title=None, axis=alt.Axis(format="%H:%M"), scale=alt.Scale(domain=domain)),
        x2="end:T",
        y=alt.Y("name:N", title=None),
        tooltip=["name", alt.Tooltip("start:T", format="%H:%M"), alt.Tooltip("end:T", format="%H:%M")],
    )
    rule = alt.Chart(pd.DataFrame({"now": _local([now])})).mark_rule(
        color=BLUE, strokeWidth=2).encode(x="now:T")
    return alt.layer(bars, rule).properties(height=40 + 34 * len(df))


def consumption_chart(meter):
    """kWh charged per night and van over the last days (stacked bars)."""
    return alt.Chart(meter).mark_bar().encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b")),
        y=alt.Y("sum(kwh):Q", title="kWh per night"),
        color=alt.Color("name:N", title="Van", scale=alt.Scale(range=VAN_COLOURS)),
        tooltip=["name", alt.Tooltip("date:T", format="%a %d %b"), "kwh"],
    ).properties(height=260)


def actual_vs_predicted(test):
    """Model page: actual and predicted price over the test period."""
    test = test.assign(ts=_local(test["ts"]).values)
    long = test.melt(id_vars="ts", value_vars=["actual", "predicted"],
                     var_name="series", value_name="rp_kwh")
    return alt.Chart(long).mark_line().encode(
        x=alt.X("ts:T", title=None),
        y=alt.Y("rp_kwh:Q", title="Rp./kWh"),
        color=alt.Color("series:N", title=None, scale=alt.Scale(range=[LINE, BLUE])),
    ).properties(height=280)
