"""
planer.py - STUB (placeholder) for role C (ML & Planner).

Generated with Claude (Anthropic) on 19 Nov 2026 so that Cosmo's pages can
call the agreed interface before C's real planner exists. It is a simple
greedy version. C replaces it with the real, tested one and keeps the names:

    plan_ev_charging(battery_kwh, soc_now, soc_min, min_by, soc_target,
                     power_kw, prices) -> list of (ts, kwh)
    cost_chf(blocks, prices) -> float
    cost_on_arrival_chf(energy_kwh, power_kw, prices) -> float
"""

import pandas as pd


def _fill(order, need, room, plan):
    """Put up to `need` kWh into the hours of `order`, cheapest first.

    `room` says how many kWh each hour can still take (charger limit),
    `plan` collects the kWh per hour. Returns the kWh that did not fit.
    """
    for ts in order["ts"]:
        if need <= 1e-9:
            break
        take = min(room[ts], need)
        if take > 0:
            room[ts] -= take
            plan[ts] = plan.get(ts, 0.0) + take
            need -= take
    return need


def plan_ev_charging(battery_kwh, soc_now, soc_min,
                     min_by, soc_target, power_kw,
                     prices):
    """Plan when one van charges.

    Args:
        battery_kwh: battery size, e.g. 80.
        soc_now, soc_min, soc_target: charge levels as fractions (0.25 = 25 %).
        min_by: datetime by which soc_min must be reached (departure).
        power_kw: charger power, i.e. max kWh per hour.
        prices: DataFrame with ts and rp_kwh, ONLY the hours the van is
            plugged in (tonight and, if wanted, the next night).

    Returns:
        List of (ts, kwh) sorted by time.

    Raises:
        ValueError: if the minimum cannot be reached before `min_by`.
    """
    room = {ts: float(power_kw) for ts in prices["ts"]}
    plan = {}

    # Step 1: the minimum for tomorrow's route, in the cheapest hours before departure
    need_min = max(0.0, battery_kwh * (soc_min - soc_now))
    before = prices[prices["ts"] < pd.Timestamp(min_by)].sort_values("rp_kwh")
    missing = _fill(before, need_min, room, plan)
    if missing > 1e-6:
        raise ValueError(f"cannot reach {soc_min:.0%} by {pd.Timestamp(min_by):%H:%M} "
                         f"({missing:.1f} kWh missing). Lower the minimum or use a faster charger.")

    # Step 2: the rest up to the target, in the cheapest hours of the whole horizon
    rest = max(0.0, battery_kwh * (soc_target - soc_now)) - sum(plan.values())
    _fill(prices.sort_values("rp_kwh"), rest, room, plan)

    return sorted((ts, round(kwh, 2)) for ts, kwh in plan.items() if kwh > 0)


def cost_chf(blocks, prices):
    """Cost in CHF of a plan, using the price of each hour."""
    price_of = dict(zip(prices["ts"], prices["rp_kwh"]))
    return round(sum(kwh * price_of[ts] for ts, kwh in blocks) / 100, 2)


def cost_on_arrival_chf(energy_kwh, power_kw, prices):
    """Cost in CHF if the van simply charges from the moment it is plugged in."""
    rappen, need = 0.0, energy_kwh
    for _, hour in prices.sort_values("ts").iterrows():
        if need <= 0:
            break
        take = min(power_kw, need)
        rappen += take * hour["rp_kwh"]
        need -= take
    return round(rappen / 100, 2)
