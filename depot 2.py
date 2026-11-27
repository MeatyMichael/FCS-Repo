"""
depot.py - the simulated depot, light version (role B, Cosmo).

This is our answer to the professor's "simulate the connection": no real
chargers, but a simulated clock and a status per vehicle.

make_plan() and soc_at() were generated with Claude (Anthropic) on
19 Nov 2026, status_at() and changes() on 27 Nov 2026.
Test: `pytest tests/test_depot.py`.
Bonus (role D, only if time): classes Van and Depot with a tick() method.
"""

from datetime import timedelta


def make_plan(name, blocks, battery_kwh, soc_now,
              power_kw, plug_in, departure):
    """Turn tonight's charging hours of one van into one plan dictionary.

    Simplification: the van charges from its first to its last planned hour
    tonight, without gaps. That is close enough for the simulation.

    Returns e.g. {"name": "Van 1", "start": 02:00, "end": 04:11,
                  "soc_start": 25, "soc_end": 55}
    """
    tonight = [(ts, kwh) for ts, kwh in blocks if ts < departure]
    kwh_tonight = sum(kwh for _, kwh in tonight)
    if tonight:
        start = tonight[0][0]
        last_ts, last_kwh = tonight[-1]
        end = last_ts + timedelta(hours=last_kwh / power_kw)
    else:  # nothing to charge tonight: the van is ready from the start
        start = end = plug_in
    return {
        "name": name,
        "start": start,
        "end": end,
        "soc_start": round(soc_now),
        "soc_end": round(soc_now + kwh_tonight / battery_kwh * 100),
    }


def soc_at(plan, now):
    """Charge level (%) of the van at time `now`, assuming steady charging."""
    if now <= plan["start"]:
        return plan["soc_start"]
    if now >= plan["end"]:
        return plan["soc_end"]
    share = (now - plan["start"]) / (plan["end"] - plan["start"])
    return plan["soc_start"] + share * (plan["soc_end"] - plan["soc_start"])


def status_at(plans, now):
    """Say what each van is doing at time `now`.

    Args:
        plans: list of dictionaries like
            {"name": "Van 1", "start": datetime, "end": datetime, ...}
        now: the simulated time (a datetime)

    Returns:
        A dictionary name -> status, with status one of
            "scheduled"  if now is before start
            "charging"   if start <= now < end
            "ready"      if now is at or after end

    Example (start 02:00, end 04:00):
        at 01:00 -> {"Van 1": "scheduled"}
        at 03:00 -> {"Van 1": "charging"}
        at 05:00 -> {"Van 1": "ready"}
    """
    result = {}
    for plan in plans:
        # Datetimes can be compared with < just like numbers
        if now < plan["start"]:
            result[plan["name"]] = "scheduled"
        elif now < plan["end"]:
            result[plan["name"]] = "charging"
        else:
            result[plan["name"]] = "ready"
    return result


def changes(plans, old_now, new_now):
    """Notification texts for every van whose status changed.

    Example: Van 1 goes from "scheduled" at 01:00 to "charging" at 02:00
        -> ["Van 1 started charging"]
    Each message must contain the van's name. A van whose status did not
    change gets no message.
    """
    before = status_at(plans, old_now)   # e.g. {"Van 1": "scheduled"}
    after = status_at(plans, new_now)    # e.g. {"Van 1": "charging"}
    # The text for each new status
    words = {"charging": "started charging", "ready": "is ready", "scheduled": "is scheduled"}
    messages = []
    for name in after:
        if after[name] != before.get(name):   # the status changed in this hour
            messages.append(f"{name} {words[after[name]]}")
    return messages