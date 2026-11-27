"""
modell.py - STUB (placeholder) for role C (ML).

Generated with Claude (Anthropic) on 19 Nov 2026 so that the Model page has
numbers to show. THE NUMBERS ARE FAKE. C replaces this file with the real
model (features, time-based split, linear regression vs the baseline
"same hour last week") and keeps the function name and the dictionary keys.
"""

from datetime import datetime, timedelta

import numpy as np

from daten import TZ, get_market_prices


def get_model_results():
    """Return the test results of the price model.

    Keys:
        baseline_mae: MAE of "same hour last week" in Rp./kWh
        model_mae: MAE of our model in Rp./kWh
        test: DataFrame with ts, actual, predicted (exchange price in Rp./kWh,
              without the tariff surcharge, like the table ml_data)
        placeholder: True only in this stub (the page then says so).
              C's real version leaves it out.
    """
    # Fake test period: the last 14 days of real exchange prices
    end = datetime.now(TZ).replace(minute=0, second=0, microsecond=0) - timedelta(days=1)
    start = end - timedelta(days=14)
    test = get_market_prices(start, end)[["ts", "market_rp"]].rename(columns={"market_rp": "actual"})
    # Fake "prediction": the real price plus random noise (seed 7, so it stays the same)
    rng = np.random.default_rng(7)
    test["predicted"] = (test["actual"] + rng.normal(0, 2.6, len(test))).round(2)
    model_mae = float((test["actual"] - test["predicted"]).abs().mean())
    return {"baseline_mae": 3.4, "model_mae": round(model_mae, 2), "test": test, "placeholder": True}
