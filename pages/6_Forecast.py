"""Page 6 · Forecast - how good is our price forecast?"""
"""
Page 6 · Forecast - how good is our price forecast? (role B shows it, C computes it)

Generated with Claude (Anthropic) on 19 Nov 2026 as a starting skeleton,
changed on 23 and 27 Nov 2026 to match the team database (ml_data, predictions)
and on 27 Nov 2026 for the new look.
The numbers come from modell.get_model_results(), a stub until C's model is ready.
"""

import streamlit as st

import grafiken
from modell import get_model_results


# ttl=3600: keep the result for one hour, then compute it again (new prices come in)
@st.cache_data(ttl=3600)
def load_results():
    """Cached, so the model results are not recomputed on every click."""
    return get_model_results()


results = load_results()

st.title("Forecast")
# The answer first, in plain words
st.markdown(f"Our model misses the real price by **{results['model_mae']:.1f} Rp./kWh** on average. "
            f"The simple rule “same as last week” misses by {results['baseline_mae']:.1f} Rp./kWh.")
if results.get("placeholder"):
    st.info("These numbers are placeholders until C's model is ready (modell.py).")
# Three tabs, so the page stays short: numbers, what the forecast is used for, method
accuracy, forecast, how = st.tabs(["Accuracy", "Tonight and after", "How it works"])

#our model against the simple rule (grading criterion 5: ML) 
with accuracy:
    c1, c2, c3 = st.columns(3)
    c1.metric("Simple rule, average error", f"{results['baseline_mae']:.2f} Rp./kWh",
              help="Rule: the price is the same as in the same hour last week.")
    c2.metric("Our model, average error", f"{results['model_mae']:.2f} Rp./kWh")
    c3.metric("Improvement", f"{results['baseline_mae'] - results['model_mae']:.2f} Rp./kWh",
              help="How much closer our model is to the real price, on average.")
    st.altair_chart(grafiken.actual_vs_predicted(results["test"]), width="stretch")
    st.caption("Test period: weeks the model never saw during training (split by time, not shuffled).")

 #where the forecast is used
with forecast:
    st.write("The exchange publishes prices only until tomorrow night. For the night after, "
             "the Plan page uses C's forecast from the table predictions in electricity_prices.db. "
             "While that table is empty, it takes the price of the same hour one week earlier.")

#the method in plain words (
with how:
    st.markdown(
        """
- **Question:** can we estimate the exchange price for the day after tomorrow (day 2) and the day after that (day 3)?
- **Data:** hourly prices for Switzerland and Germany since January 2024 (Energy-Charts) and weather
  (Open-Meteo), all in the table ml_data.
- **Features:** hour, weekday, weekend, month, public holidays in CH and DE, the Swiss and German price
  1, 2 and 7 days earlier, and the weather forecast (temperature, wind and sun in Switzerland,
  wind at three places in northern Germany, where the big wind farms are).
- **No data leakage:** the model only uses what is known at 13:00 today. Two models: day 2 may use
  tomorrow's price (24 h earlier), day 3 may not. Training uses the weather as it was *forecast*
  2 or 3 days ahead, not as it was measured, so the test is honest.
- **Check:** split by time (the last months are the test), compared with the rule
  “same hour last week” (MAE in Rp./kWh).
- **Why it matters:** tonight's prices are known, the night after is not. The forecast fills that gap.
"""
    )