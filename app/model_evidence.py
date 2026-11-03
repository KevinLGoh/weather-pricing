"""Model Evidence: out-of-sample scores, confidence intervals and where they come from."""

import pandas as pd
import streamlit as st

import shared
from weather_pricing import inference
from weather_pricing.experiment import MAIN

CONCLUSION = (
    "What this supports: for this weather baseline (one archived deterministic NWP run, Open-Meteo "
    "best match, with simple statistical post-processing), the Kalshi prices held the information in "
    "the forecast, and adding the forecast to the prices did not improve them. This is not evidence "
    "that prediction markets are better than meteorology in general: traders very probably use "
    "ensembles, NBM guidance and live station observations that this baseline does not have."
)

LIMITATIONS = [
    "Forecast age: the weather models only use forecast runs started at least a day before the hour "
    "they forecast, so at the snapshot the forecast is often 7–23 h old, and it is the same at "
    "D-0 08:00 and D-0 12:00. This likely overstates the market's advantage.",
    "Short horizons: these markets cover 34 h or less, so the results say nothing about weather risk "
    "over many days or seasons.",
    "Weak weather baseline: one deterministic forecast run without ensemble spread or intraday "
    "observations; traders very probably have more.",
    "Only one chronological train/test split: no rolling-origin evaluation yet.",
    "Stale quotes: in illiquid hours the last hourly quote carries over; such rows are flagged stale and "
    "analysed separately.",
    "Renormalised quotes: at a snapshot where not every bracket of an event was quoted, the raw mid is "
    "kept instead of the renormalised one.",
    "Settled temperatures before 2025: Kalshi mostly omits them, so NOAA values fill the gap, only for "
    "the rule-based error model; outcomes always come from Kalshi.",
    "The note is hypothetical, not a traded security, and the risk loading λ is illustrative.",
    "New York only.",
]

BRIER_COLUMNS = {
    "A_xgb": "Weather-only, XGBoost",
    "B_market_norm": "Market mid, renormalised",
    "C_logit": "Combined, logistic",
    "C_xgb": "Combined, XGBoost",
    "A_minus_C_xgb": "Gain from adding the market (XGBoost)",
}
CALIBRATION_MODELS = ["A_xgb", "B_market_norm", "C_xgb"]
# Liquidity buckets as experiment.run labels them, tightest spread first.
LIQUIDITY_ORDER = ["spread<=2c", "2-5c", ">5c", "stale/no quote"]
# The probabilities experiment.run passes to the encompassing regression.
ENCOMPASSING_MODELS = {"weather": "A_logit", "market": "B_market_norm"}

ENCOMPASSING_TERMS = {
    "const": "Intercept",
    "logit_weather": "Weather probability (logit)",
    "logit_market": "Market probability (logit)",
}

shared.page_header(
    "Model Evidence",
    [
        "All results are out-of-sample: each model was fitted on earlier days and scored on the later "
        "test days only.",
        "Lower is better for the Brier score, the log loss and the ECE (expected calibration error).",
        "In the bootstrap table a positive difference means the challenger has the lower (better) Brier "
        "score; the 95% interval comes from resampling whole city-days.",
    ],
)


def p_text(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def brier_table(rows: pd.DataFrame, first: str, first_column: str) -> pd.DataFrame:
    """Brier scores of one robustness cut, as display strings."""
    table = {first_column: list(rows[first]), "City-days": [int(n) for n in rows["events"]]}
    for column, label in BRIER_COLUMNS.items():
        table[label] = [f"{x:.4f}" for x in rows[column]]
    return pd.DataFrame(table)


with shared.guarded():
    evidence = inference.load_evidence()
    split, manifest = evidence["split"], evidence["manifest"]
    test = split["test"]
    st.caption(
        f"Test period {test['dates'][0]} to {test['dates'][1]}: {test['events']} days, "
        f"{test['rows']:,} contract snapshots."
    )

    st.subheader("Scores")
    scores = evidence["scores"].reindex(MAIN)
    st.dataframe(
        pd.DataFrame(
            {
                "Model": [shared.MODEL_LABELS[m] for m in MAIN],
                "Brier": [f"{x:.4f}" for x in scores["brier"]],
                "Log loss": [f"{x:.3f}" for x in scores["log_loss"]],
                "ECE": [f"{x:.3f}" for x in scores["ece"]],
            }
        ),
        hide_index=True,
    )

    st.subheader("Does one source add to the other?")
    boot = evidence["bootstrap"]
    st.dataframe(
        pd.DataFrame(
            {
                "Question": boot["question"],
                "Baseline": [shared.MODEL_LABELS[m] for m in boot["baseline"]],
                "Challenger": [shared.MODEL_LABELS[m] for m in boot["challenger"]],
                "Brier difference": [f"{d:.4f}" for d in boot["diff"]],
                "95% CI": [f"{lo:.4f} to {hi:.4f}" for lo, hi in zip(boot["ci_lo"], boot["ci_hi"])],
                "p-value": [p_text(p) for p in boot["p_value"]],
                "City-days": [int(n) for n in boot["n_events"]],
            }
        ),
        hide_index=True,
    )

    st.subheader("Encompassing regression")
    enc = evidence["encompassing"].reindex(list(ENCOMPASSING_TERMS))
    weather_p = float(enc.loc["logit_weather", "p_value"])
    p_short = "< 0.01" if weather_p < 0.01 else f"= {weather_p:.2f}"
    if weather_p < 0.05:  # the sentence must follow the data, whichever way it comes out
        weather_reading = f"is clearly different from zero (p {p_short})"
    else:
        weather_reading = (
            f"is not clearly different from zero (p {p_short}), so the weather model adds no clear "
            "information once the market is known"
        )
    st.markdown(
        "A logistic regression of the outcome on both probabilities at once: the weather probability of "
        f"{shared.MODEL_LABELS[ENCOMPASSING_MODELS['weather']]} and the market probability of "
        f"{shared.MODEL_LABELS[ENCOMPASSING_MODELS['market']]}. A clearly non-zero market coefficient means "
        f"the market has information the weather model lacks. The weather coefficient {weather_reading}."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Term": list(ENCOMPASSING_TERMS.values()),
                "Coefficient": [f"{x:.2f}" for x in enc["coef"]],
                "SE": [f"{x:.2f}" for x in enc["se"]],
                "p-value": [p_text(p) for p in enc["p_value"]],
            }
        ),
        hide_index=True,
    )

    st.info(CONCLUSION)

    with st.expander("Where these results come from"):
        train = split["train"]
        lines = [
            f"Data: {manifest['data_date_range'][0]} to {manifest['data_date_range'][1]}, "
            f"cities {', '.join(manifest['cities'])}.",
            f"Training period: {train['dates'][0]} to {train['dates'][1]} ({train['events']} days).",
            f"Test period: {test['dates'][0]} to {test['dates'][1]} ({test['events']} days).",
            f"Rows used: {split['rows_used']:,} of {split['rows_in']:,} snapshot rows "
            "(the others lack a complete forecast day, a quote or a settlement value).",
            f"Code: commit {manifest['git_commit'][:12]}, source tree {manifest['src_tree'][:12]}.",
            "Packages: " + ", ".join(f"{name} {version}" for name, version in manifest["packages"].items()) + ".",
        ]
        quality = evidence["data_quality"]
        if quality:
            lines.append(
                f"Data quality: bracket logic agrees with Kalshi for {quality['bracket_logic_matches_result']:.2%} "
                f"of contracts, NOAA for {quality['noaa_outcome_matches_result']:.2%}; "
                f"{quality['missing_forecast_rows']:,} rows without a complete forecast day; "
                f"{quality['incomplete_bracket_rows']:,} rows with an incomplete bracket set."
            )
        st.markdown("\n".join(f"- {line}" for line in lines))

    st.subheader("Calibration")
    st.markdown(
        "Contracts are grouped by predicted probability; a well-calibrated model's observed frequency "
        "matches its prediction, so its line follows the diagonal."
    )
    reliability = evidence["reliability"]
    reliability = reliability[reliability["model"].isin(CALIBRATION_MODELS)]
    labels = [shared.MODEL_LABELS[m] for m in reliability["model"]]
    chart = pd.DataFrame(
        {
            "Mean predicted probability": list(reliability["mean_pred"]) + [0.0, 1.0],
            "Observed frequency": list(reliability["obs_freq"]) + [0.0, 1.0],
            "Model": labels + ["Perfect calibration"] * 2,
        }
    )
    # x and y as columns: with an index, Streamlit 1.66 triggers a pandas 3.1 deprecation warning.
    st.line_chart(chart, x="Mean predicted probability", y="Observed frequency", color="Model")
    st.dataframe(
        pd.DataFrame(
            {
                "Model": labels,
                "Bin": [f"{lo:.1f}–{lo + 0.1:.1f}" for lo in reliability["bin_lo"]],
                "Contracts": [int(n) for n in reliability["n"]],
                "Mean predicted": [f"{x:.3f}" for x in reliability["mean_pred"]],
                "Observed": [f"{x:.3f}" for x in reliability["obs_freq"]],
            }
        ),
        hide_index=True,
    )

    robustness = evidence["robustness"]
    st.subheader("By snapshot")
    lead = robustness[robustness["cut"] == "lead"].set_index("snapshot")
    lead = lead.reindex([s for s in inference.SNAPSHOT_LABELS if s in lead.index]).reset_index()
    st.dataframe(brier_table(lead, "snapshot", "Snapshot"), hide_index=True)

    with st.expander("Robustness by liquidity and settlement source"):
        liquidity = robustness[robustness["cut"] == "liquidity"]
        rank = {label: i for i, label in enumerate(LIQUIDITY_ORDER)}
        liquidity = liquidity.sort_values("liquidity", key=lambda s: s.map(rank).fillna(len(rank)), kind="stable")
        st.dataframe(brier_table(liquidity, "liquidity", "Liquidity"), hide_index=True)
        sources = robustness[robustness["cut"] == "settle_source"]
        st.dataframe(brier_table(sources, "settle_source", "Settlement source"), hide_index=True)
        twc = sources[sources["settle_source"] == "The Weather Company"]
        if not twc.empty:
            st.caption(
                f"The Weather Company settles {int(twc['events'].iloc[0])} of {test['events']} test city-days, "
                "so its numbers are less certain."
            )

    st.subheader("Limitations")
    st.markdown("\n".join(f"- {line}" for line in LIMITATIONS))

    st.subheader("Data sources")
    st.markdown(shared.data_sources(manifest), unsafe_allow_html=True)
