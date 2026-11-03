"""Contract Explorer: pick a city, day, snapshot and contract."""

import streamlit as st

import shared
from weather_pricing import inference

shared.page_header(
    "Contract Explorer",
    [
        "Pick a city, a day in the test period, a snapshot time and a contract (a temperature bracket).",
        "The page shows the probability each model gave the bracket, and what really happened.",
        "Snapshots are local times: D-1 is the day before the target day, D-0 the day itself.",
    ],
)

with shared.guarded():
    city = st.selectbox("City", inference.list_cities())
    dates = sorted(inference.list_dates(city), reverse=True)
    st.caption(f"Loaded {len(dates)} test days ({dates[-1]} to {dates[0]}) from the app artifacts.")
    day = st.selectbox("Date", dates, format_func=lambda d: d.isoformat())
    snapshot = st.selectbox("Snapshot", inference.list_snapshots(city, day))
    contracts = inference.list_contracts(city, day, snapshot)
    label = st.selectbox("Contract", contracts["label"].tolist())
    st.dataframe(
        contracts[["label", "bid", "ask", "mid"]].rename(
            columns={"label": "Bracket", "bid": "Bid", "ask": "Ask", "mid": "Mid"}
        ),
        hide_index=True,
    )
