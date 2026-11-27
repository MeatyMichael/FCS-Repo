"""Savings page"""

import streamlit as st

st.title("Savings")
st.info("Bonus page. We build it in Phase 2 only if everything else works.")
st.markdown(
    """
**What it will show**

- The charging plan replayed on the last 12 months of real prices
- Savings per month: charging on arrival vs our plan with the forecast vs perfect knowledge
- The share of the possible saving that the forecast gets (proof that the ML helps)
- Heatmap: average price by hour and weekday
"""
)