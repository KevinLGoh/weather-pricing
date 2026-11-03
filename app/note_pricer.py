"""Note Pricer: value a hypothetical note on the chosen contract under every model."""

import pandas as pd
import streamlit as st

import shared
from weather_pricing import inference

LAMBDA_CAPTION = (
    "λ is illustrative: a Wang-transform risk loading chosen for this example, not calibrated to any market."
)

# Worded to match valuation.py: V = D * (C + N * (1 - q * L)).
shared.page_header(
    "Note Pricer",
    [
        "A hypothetical one-period note on the contract you picked. At settlement it pays the principal "
        "plus the coupon.",
        "If the daily high lands in the bracket (the contract settles Yes), it loses the loss fraction of "
        "the principal; the coupon is paid either way.",
        "Each value is the present value at the snapshot, discounted at the evaluation's rate, using the "
        "model's probability (after the λ loading) as the chance of the loss. Change the inputs below.",
    ],
)


def money(x: float) -> str:
    return f"${x:,.2f}"


def signed_money(x: float) -> str:
    return "$0.00" if abs(x) < 0.005 else f"{'+' if x > 0 else '-'}{money(abs(x))}"


def back_to_explorer(message: str) -> None:
    st.info(message)
    st.page_link("contract_explorer.py", label="Go to the Contract Explorer")
    st.stop()


selection = st.session_state.get("selection")
if selection is None:
    back_to_explorer("Pick a contract in the Contract Explorer first.")

with shared.guarded():
    try:
        result = inference.predict(selection["city"], selection["date"], selection["snapshot"], selection["ticker"])
    except (inference.DataNotFoundError, inference.InvalidInputError):
        # The stored choice is from older artifacts; artifact problems still reach guarded().
        back_to_explorer(
            "The contract you picked is no longer in the app's data. Pick one again in the Contract Explorer."
        )
    contract = result["contract"]
    st.markdown(
        f"Contract: {selection['city']}, {selection['date']}, {selection['snapshot']}, {contract['label']}."
    )

    # "\$": Streamlit Markdown (also in widget labels) reads a $...$ pair as inline math.
    principal = st.number_input("Principal (\\$)", min_value=1.0, max_value=1e9, value=1000.0, step=100.0)
    loss_pct = st.slider("Loss if the bracket happens (% of principal)", min_value=0, max_value=100, value=50)
    coupon = st.number_input("Coupon (\\$, paid at settlement)", min_value=0.0, max_value=1e9, value=0.0, step=10.0)
    lam = st.slider("Risk loading λ", min_value=-1.0, max_value=1.0, value=0.0, step=0.05)
    st.caption(LAMBDA_CAPTION)
    models = list(result["probabilities"])
    baseline_label = st.selectbox(
        "Baseline model", [shared.MODEL_LABELS[m] for m in models], index=models.index("A_xgb")
    )
    baseline = models[[shared.MODEL_LABELS[m] for m in models].index(baseline_label)]

    prices = inference.price_note(
        result["probabilities"],
        contract["lead_hours"],
        principal=principal,
        loss_fraction=loss_pct / 100,  # the slider shows % of principal; valuation uses a fraction
        coupon=coupon,
        lam=lam,
        baseline=baseline,
    )
    labels = [shared.MODEL_LABELS[m] for m in prices["model"]]
    st.dataframe(
        pd.DataFrame(
            {
                "Model": labels,
                "Family": [shared.FAMILY_LABELS[f] for f in prices["family"]],
                "Probability": [f"{p:.1%}" for p in prices["probability"]],
                "Pricing probability": [f"{p:.1%}" for p in prices["pricing_probability"]],
                "Value": [money(v) for v in prices["value"]],
                "Change vs baseline": [signed_money(c) for c in prices["change_vs_baseline"]],
            }
        ),
        hide_index=True,
    )
    st.bar_chart(
        pd.DataFrame({"Model": labels, "Change vs baseline ($)": list(prices["change_vs_baseline"])}),
        x="Model",
        y="Change vs baseline ($)",
        horizontal=True,
        sort=False,
    )
    if baseline != "C_xgb":
        change = float(prices.set_index("model")["change_vs_baseline"]["C_xgb"])
        st.markdown(shared.comparison_sentence(change, baseline_label))
