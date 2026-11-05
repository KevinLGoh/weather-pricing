"""Contract Explorer: pick a city, day, snapshot and contract; compare the models."""

import pandas as pd
import streamlit as st

import shared
from weather_pricing import inference
from weather_pricing.config import CITIES, STALE_HOURS

LIMITATION = (
    "Known limitation: the weather models only use forecast runs started at least a day before "
    "the hour they forecast (and published about 6 h after they start), so at the snapshot the "
    "forecast is often 7–23 h old, and it is the same at D-0 08:00 and D-0 12:00. This likely "
    "overstates the market's advantage."
)
STRONGER_LIMITATION = (
    "Weather-only models use Open-Meteo forecasts, which are often 7–23 h old at the snapshot. Stronger "
    "weather adds NWS guidance, usually 2–10 h old, and the readings so far that day. Extended weather adds "
    "NAM and three global models. None of them has hourly guidance, so a live forecaster could do better."
)

shared.page_header(
    "Contract Explorer",
    [
        "Pick a city, a day in the test period, a snapshot time and a contract (a temperature bracket).",
        "The page shows the probability each model gave the bracket, and what really happened.",
        "Snapshots are local times: D-1 is the day before the target day, D-0 the day itself.",
    ],
)

with shared.guarded():
    cities = inference.list_cities()
    # New York first: it is the longest-studied city and the README's worked example.
    city = st.selectbox("City", cities, index=cities.index("NY") if "NY" in cities else 0)
    if city in CITIES:
        st.caption(f"Settles at {CITIES[city].name}.")
    dates = sorted(inference.list_dates(city), reverse=True)
    st.caption(f"Loaded {len(dates)} test days ({dates[-1]} to {dates[0]}) from the app artifacts.")
    day = st.selectbox("Date", dates, format_func=lambda d: d.isoformat())
    snapshot = st.selectbox("Snapshot", inference.list_snapshots(city, day))
    contracts = inference.list_contracts(city, day, snapshot)
    label = st.selectbox("Contract", contracts["label"].tolist())
    # Dollar prices with two decimals, like the quote line below; "—" where there is no quote.
    def dollars(x) -> str:
        return "—" if pd.isna(x) else f"${x:.2f}"

    st.dataframe(
        pd.DataFrame(
            {
                "Bracket": contracts["label"],
                "Bid": [dollars(x) for x in contracts["bid"]],
                "Ask": [dollars(x) for x in contracts["ask"]],
                "Mid": [dollars(x) for x in contracts["mid"]],
            }
        ),
        hide_index=True,
    )

    ticker = contracts.loc[contracts["label"] == label, "ticker"].iloc[0]
    result = inference.predict(city, day, snapshot, ticker)
    # The Note Pricer picks up this choice.
    st.session_state["selection"] = {"city": city, "date": day.isoformat(), "snapshot": snapshot, "ticker": ticker}

    quote = result["contract"]
    # Prices are dollars per $1 contract. Without a two-sided quote the dataset's "mid" is
    # the last trade price (dataset.market_state), so it is named that way here.
    # "\$": Streamlit Markdown reads a $...$ pair as inline math, so every $ is escaped.
    if quote["mid"] is None:
        st.markdown("Market quote: none at this snapshot.")
    elif quote["spread"] is None:
        st.markdown(f"Market quote: last trade price \\${quote['mid']:.2f} (no two-sided quote at this snapshot).")
    else:
        st.markdown(
            f"Market quote: mid price \\${quote['mid']:.2f} (bid \\${quote['bid']:.2f}, "
            f"ask \\${quote['ask']:.2f}, spread \\${quote['spread']:.2f})."
        )
    if quote["stale"]:
        st.warning(
            f"This quote is stale: no trade in the {STALE_HOURS:.0f} hours before the snapshot, "
            "or no two-sided quote at the snapshot, so the market price may be out of date."
        )

    probs = result["probabilities"]
    for family, models in result["families"].items():
        st.subheader(shared.FAMILY_LABELS[family])
        for column, model in zip(st.columns(len(models)), models):
            column.metric(shared.MODEL_LABELS[model], f"{probs[model]:.1%}")
    # Model names as a column with x= (not the index): with an index, Streamlit 1.66 calls
    # reset_index(inplace=True), which pandas 3.1 deprecates.
    st.bar_chart(
        pd.DataFrame({"Model": [shared.MODEL_LABELS[m] for m in probs], "Probability": list(probs.values())}),
        x="Model",
        y="Probability",
        horizontal=True,
        sort=False,  # keep the family order of the metrics above
    )
    st.caption(STRONGER_LIMITATION if "stronger_weather" in result["families"] else LIMITATION)

    outcome = result["outcome"]
    settled = "unknown" if outcome["settled_value"] is None else f"{outcome['settled_value']:.0f}°F"
    paid = "Yes" if outcome["y"] else "No"
    (st.success if outcome["y"] else st.info)(f"Outcome: the daily high was {settled}, so this contract paid {paid}.")

if st.button("Price this contract", type="primary"):
    st.switch_page("note_pricer.py")
