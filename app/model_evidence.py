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

FINDING_CONCLUSION = (
    "What this supports: in liquid Kalshi daily-high markets for New York and Chicago, prices beat every "
    "public-archive weather model we could build, and those models add almost nothing to prices out of "
    "sample. This does not show that markets beat a live forecaster with hourly data, which the archives "
    "do not offer, and it says nothing about other weather events or longer horizons."
)

# Replace the single-run limitations about the weather baseline, the split and the city.
FINDING_LIMITATIONS = [
    "Guidance age: the stronger models use guidance usually 2–10 h old. Individual runs exist only from "
    "April 2026 and are 3- to 6-hourly, and full ensembles are not archived, so part of the market's lead may "
    "still be a data gap, especially on the target day.",
    "Release delays are assumptions, not archived publication times. Doubling the first set of guidance "
    "delays did not change the conclusions.",
    "Short horizons: these markets cover 34 h or less, so the results say nothing about weather risk "
    "over many days or seasons.",
    "Two cities, which share weather on some days. The per-city results use one chronological "
    "train/test split, and the rolling-origin tables use 5 windows.",
    "Stale quotes: in illiquid hours the last hourly quote carries over. Such rows are flagged stale and "
    "analysed separately.",
    "Strike placement: Kalshi sets its brackets near its expected temperature, so even the weather-only "
    "models see a small market signal in the contract terms.",
    "Settled temperatures before 2025: Kalshi mostly omits them, so NOAA values fill the gap, only for "
    "the rule-based error model. Outcomes always come from Kalshi.",
    "The note is hypothetical, not a traded security, and the risk loading λ is illustrative.",
]

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
    "W_xgb": "Stronger weather, XGBoost",
    "X_xgb": "Extended weather, XGBoost",
    "B_market_norm": "Market mid, renormalised",
    "C_logit": "Combined, logistic",
    "C_xgb": "Combined, XGBoost",
    "A_minus_C_xgb": "Gain from adding the market (XGBoost)",
}
CALIBRATION_MODELS = ["A_xgb", "W_xgb", "X_xgb", "B_market_norm", "C_xgb"]
# Liquidity buckets as experiment.run labels them, tightest spread first.
LIQUIDITY_ORDER = ["spread<=2c", "2-5c", ">5c", "stale/no quote"]
LIQUIDITY_LABELS = {
    "spread<=2c": "Spread ≤ 2¢",
    "2-5c": "Spread 2–5¢",
    ">5c": "Spread > 5¢",
    "stale/no quote": "Stale or no quote",
}
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
        "test days only. The exact-run blend is the one exception, and its section explains how.",
        "Lower is better for the Brier score, the log loss and the ECE (expected calibration error).",
        "In the bootstrap table a positive difference means the challenger has the lower (better) Brier "
        "score; the 95% interval comes from resampling whole city-days.",
    ],
)


def p_text(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def p_with_equals(p: float) -> str:
    """For use after the letter p in a sentence: 'p < 0.001' or 'p = 0.154'."""
    return p_text(p) if p < 0.001 else f"= {p_text(p)}"


def brier_table(rows: pd.DataFrame, first: str, first_column: str) -> pd.DataFrame:
    """Brier scores of one robustness cut, as display strings."""
    table = {first_column: list(rows[first]), "City-days": [int(n) for n in rows["events"]]}
    for column, label in BRIER_COLUMNS.items():
        if column in rows:  # stronger weather models are only in the multi-city build
            table[label] = [f"{x:.4f}" for x in rows[column]]
    return pd.DataFrame(table)


def fixed(x: float, digits: int = 4, signed: bool = False) -> str:
    """`x` with `digits` decimals, for differences and interval bounds.

    A small value that would round to zero keeps further decimals until it shows, so a table
    never has "-0.0000": the sign says which side of zero a bound is on, and the digits after
    it must show that too. `signed` puts a plus sign on positive values.
    """
    for decimals in range(digits, digits + 4):
        text = f"{x:.{decimals}f}"
        if float(text) != 0:
            break
    else:
        text = f"{0:.{digits}f}"
    return f"+{text}" if signed and float(text) > 0 else text


def interval(lo: float, hi: float, digits: int = 3) -> str:
    return f"{fixed(lo, digits)} to {fixed(hi, digits)}"


def show_finding(finding: dict) -> None:
    """Across cities, windows and weather models: does the market's lead survive?"""
    st.subheader("Does the market's lead survive better tests?")
    st.markdown(
        "Each row gives the market's lead over the best weather model at that stage: the weather model's "
        "Brier score minus the market's. Positive means the market is better. The samples differ, so the rows "
        "are not strictly comparable."
    )
    ladder = finding["ladder"]
    shared.table(
        pd.DataFrame(
            {
                # The numbers first: on a narrow screen the columns past the edge need scrolling.
                "Test": ladder["test"],
                "Market lead": [f"{x:.3f}" for x in ladder["lead"]],
                "95% CI": [interval(lo, hi) for lo, hi in zip(ladder["ci_lo"], ladder["ci_hi"])],
                "Sample": ladder["sample"],
                "Weather model": ladder["weather_model"],
            }
        )
    )
    st.info(FINDING_CONCLUSION)

    st.subheader("Rolling-origin evaluation")
    st.markdown(
        "Each city refits on all earlier days and is scored on the next 6-month window, so a result has to hold "
        "across seasons and as the market matures. Pooled intervals resample whole dates, because two cities on "
        "one date are not independent."
    )
    # The pooled rows first: they are the headline, and each city then backs them up.
    compared = finding["rolling_comparisons"]
    compared = compared.sort_values("city", key=lambda c: (c != "ALL").astype(int), kind="stable")
    shared.table(
        pd.DataFrame(
            {
                "Sample": [shared.city_label(c) for c in compared["city"]],
                "Question": compared["question"],
                "Brier difference": [fixed(x) for x in compared["diff"]],
                "95% CI": [interval(lo, hi, 4) for lo, hi in zip(compared["ci_lo"], compared["ci_hi"])],
                "p-value": [p_text(p) for p in compared["p_value"]],
            }
        )
    )

    st.markdown("**The market's lead by window**")
    folds = finding["rolling_by_fold"]
    shared.table(
        pd.DataFrame(
            {
                "City": [shared.city_label(c) for c in folds["city"]],
                "From": folds["from"],
                "Market lead": [fixed(x) for x in folds["market_lead"]],
                "95% CI": [interval(lo, hi, 4) for lo, hi in zip(folds["lead_lo"], folds["lead_hi"])],
                "Extended weather": [f"{x:.4f}" for x in folds["X_xgb"]],
                "Market": [f"{x:.4f}" for x in folds["B_market_norm"]],
                "To": folds["to"],
                "City-days": [int(n) for n in folds["city_days"]],
            }
        )
    )

    st.markdown("**As the market matured** (both cities)")
    st.markdown(
        "Over time spreads narrowed, the market's lead grew and the weather model's extra information fell. "
        "This is five windows of different seasons, so it is suggestive and not proof of cause."
    )
    maturity = finding["rolling_by_maturity"]
    enc = finding["rolling_encompassing"]
    by_fold = enc[(enc["city"] == "ALL") & (enc["fold"] != "all")].assign(fold=lambda d: d["fold"].astype(int)).set_index("fold")
    shared.table(
        pd.DataFrame(
            {
                "Window from": maturity["from"],
                "Median spread": [f"{x * 100:.0f}¢" for x in maturity["median_spread"]],
                "Stale rows": ["< 1%" if x < 0.01 else f"{x:.0%}" for x in maturity["stale_share"]],
                "Market lead": [f"{x:.3f}" for x in maturity["market_lead"]],
                "Weather coefficient": [fixed(by_fold.loc[k, "b_weather"], 2) for k in maturity["fold"]],
                "Weather p-value": [p_text(by_fold.loc[k, "p_weather"]) for k in maturity["fold"]],
                "Market coefficient": [fixed(by_fold.loc[k, "b_market"], 2) for k in maturity["fold"]],
            }
        )
    )

    blend = finding["rolling_forward_blend"].set_index("city").loc["ALL"]
    overall = enc[(enc["city"] == "ALL") & (enc["fold"] == "all")].iloc[0]
    st.markdown(
        f"Across all windows the weather coefficient is {overall['b_weather']:.2f} (SE {overall['se_weather']:.2f}), "
        "so the market does not contain all of the forecast. That information is worth little: a blend fitted only "
        f"on earlier windows changes the Brier score by {fixed(blend['diff'], signed=True)} "
        f"(95% CI {interval(blend['ci_lo'], blend['ci_hi'], 4)}, p {p_with_equals(blend['p_value'])})."
    )

    with st.expander("By snapshot and liquidity"):
        order = {label: i for i, label in enumerate(inference.SNAPSHOT_LABELS)}
        snapshots = finding["rolling_by_snapshot"].sort_values(
            ["city", "snapshot"], key=lambda s: s.map(order) if s.name == "snapshot" else s, kind="stable"
        )
        shared.table(
            pd.DataFrame(
                {
                    "City": [shared.city_label(c) for c in snapshots["city"]],
                    "Snapshot": snapshots["snapshot"],
                    "Extended weather": [f"{x:.4f}" for x in snapshots["X_xgb"]],
                    "Market": [f"{x:.4f}" for x in snapshots["B_market_norm"]],
                    "Market lead": [fixed(x) for x in snapshots["market_lead"]],
                    "95% CI": [interval(lo, hi, 4) for lo, hi in zip(snapshots["lead_lo"], snapshots["lead_hi"])],
                }
            )
        )
        liquidity = finding["rolling_by_liquidity"]
        rank = {label: i for i, label in enumerate(LIQUIDITY_ORDER)}
        liquidity = liquidity.sort_values("liquidity", key=lambda s: s.map(rank).fillna(len(rank)), kind="stable")
        shared.table(
            pd.DataFrame(
                {
                    "Liquidity": [LIQUIDITY_LABELS.get(b, b) for b in liquidity["liquidity"]],
                    "Market lead": [fixed(x) for x in liquidity["market_lead"]],
                    "95% CI": [interval(lo, hi, 4) for lo, hi in zip(liquidity["lead_lo"], liquidity["lead_hi"])],
                    "Weather added to market": [fixed(x, signed=True) for x in liquidity["added"]],
                    "95% CI, added": [interval(lo, hi, 4) for lo, hi in zip(liquidity["added_lo"], liquidity["added_hi"])],
                }
            )
        )

    st.subheader("Exact model runs")
    st.markdown(
        "Individual HRRR, NBM and ECMWF runs exist only from April 2026, so this test blends them into the "
        "rolling-origin predictions instead of retraining. Each date block is fitted on the others, so later data "
        "helps earlier blocks, which can only favour the weather model."
    )
    exact = finding["exact_comparisons"]
    exact = exact[exact["city"] == "ALL"]
    shared.table(
        pd.DataFrame(
            {
                "Question": exact["question"],
                "Brier difference": [fixed(x) for x in exact["diff"]],
                "95% CI": [interval(lo, hi, 4) for lo, hi in zip(exact["ci_lo"], exact["ci_hi"])],
                "p-value": [p_text(p) for p in exact["p_value"]],
                "Dates": [int(n) for n in exact["n_events"]],
            }
        )
    )
    by_snapshot = finding["exact_by_snapshot"].set_index("snapshot")
    by_snapshot = by_snapshot.reindex([s for s in inference.SNAPSHOT_LABELS if s in by_snapshot.index]).reset_index()
    shared.table(
        pd.DataFrame(
            {
                "Snapshot": by_snapshot["snapshot"],
                "Weather with the runs": [f"{x:.4f}" for x in by_snapshot["F_weather"]],
                "Market": [f"{x:.4f}" for x in by_snapshot["B_market_norm"]],
                "Market lead": [fixed(x) for x in by_snapshot["market_lead"]],
                "95% CI": [interval(lo, hi, 4) for lo, hi in zip(by_snapshot["lead_lo"], by_snapshot["lead_hi"])],
            }
        )
    )


with shared.guarded():
    finding = inference.load_finding()
    if finding is not None:
        show_finding(finding)
    cities = inference.list_evidence_cities()
    if cities:
        st.subheader("Results for one city")
        city = st.selectbox("City", cities, index=cities.index("NY") if "NY" in cities else 0, format_func=shared.city_label)
    else:
        city = None
    evidence = inference.load_evidence(city)
    split, manifest = evidence["split"], evidence["manifest"]
    test = split["test"]
    st.caption(
        f"Test period {test['dates'][0]} to {test['dates'][1]}: {test['events']} city-days, "
        f"{test['rows']:,} contract snapshots."
    )

    st.subheader("Scores")
    # MAIN always, in the family order, and the stronger models when this build has them.
    scored = [m for m in inference.ALL_MODELS if m in MAIN or m in evidence["scores"].index]
    scores = evidence["scores"].reindex(scored)
    shared.table(
        pd.DataFrame(
            {
                "Model": [shared.MODEL_LABELS[m] for m in scored],
                "Brier": [f"{x:.4f}" for x in scores["brier"]],
                "Log loss": [f"{x:.3f}" for x in scores["log_loss"]],
                "ECE": [f"{x:.3f}" for x in scores["ece"]],
            }
        )
    )

    st.subheader("Does one source add to the other?")
    boot = evidence["bootstrap"]
    shared.table(
        pd.DataFrame(
            {
                # The numbers first: on a narrow screen the columns past the edge need scrolling.
                "Question": boot["question"],
                "Brier difference": [fixed(d) for d in boot["diff"]],
                "95% CI": [interval(lo, hi, 4) for lo, hi in zip(boot["ci_lo"], boot["ci_hi"])],
                "p-value": [p_text(p) for p in boot["p_value"]],
                "City-days": [int(n) for n in boot["n_events"]],
                "Baseline": [shared.MODEL_LABELS[m] for m in boot["baseline"]],
                "Challenger": [shared.MODEL_LABELS[m] for m in boot["challenger"]],
            }
        )
    )

    st.subheader("Encompassing regression")
    enc = evidence["encompassing"].reindex(list(ENCOMPASSING_TERMS))
    weather_p = float(enc.loc["logit_weather", "p_value"])
    p_short = p_text(weather_p) if weather_p < 0.001 else f"= {p_text(weather_p)}"
    if pd.isna(weather_p):
        weather_reading = "has no p-value in the results"
    elif weather_p < 0.05:  # the sentence must follow the data, whichever way it comes out
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
    shared.table(
        pd.DataFrame(
            {
                "Term": list(ENCOMPASSING_TERMS.values()),
                "Coefficient": [fixed(x, 2) for x in enc["coef"]],
                "SE": [f"{x:.2f}" for x in enc["se"]],
                "p-value": [p_text(p) for p in enc["p_value"]],
            }
        )
    )

    extended = evidence["encompassing_extended"]
    if extended is not None:
        extended = extended.reindex(list(ENCOMPASSING_TERMS))
        st.markdown(
            f"The same regression with {shared.MODEL_LABELS['X_xgb']}, on this city's test period alone. The "
            "rolling-origin table above covers much more data."
        )
        shared.table(
            pd.DataFrame(
                {
                    "Term": list(ENCOMPASSING_TERMS.values()),
                    "Coefficient": [fixed(x, 2) for x in extended["coef"]],
                    "SE": [f"{x:.2f}" for x in extended["se"]],
                    "p-value": [p_text(p) for p in extended["p_value"]],
                }
            )
        )

    if finding is None:
        st.info(CONCLUSION)

    with st.expander("Where these results come from"):
        train = split["train"]
        lines = [
            f"Data: {manifest['data_date_range'][0]} to {manifest['data_date_range'][1]}, "
            f"cities {', '.join(manifest['cities'])}.",
            f"Training period: {train['dates'][0]} to {train['dates'][1]} ({train['events']} city-days).",
            f"Test period: {test['dates'][0]} to {test['dates'][1]} ({test['events']} city-days).",
            f"Rows used: {split['rows_used']:,} of {split['rows_in']:,} snapshot rows "
            "(the others lack a complete forecast day, a quote or a settlement value).",
            f"Code: commit {manifest['git_commit'][:12]}, source tree {manifest['src_tree'][:12]}.",
            "Packages: " + ", ".join(f"{name} {version}" for name, version in manifest["packages"].items()) + ".",
        ]
        quality = evidence["data_quality"] or {}
        # Only the keys that are present: a data_quality.json from an older build lacks the newer ones.
        clauses = []
        if "bracket_logic_matches_result" in quality:
            clauses.append(f"bracket logic agrees with Kalshi for {quality['bracket_logic_matches_result']:.2%}")
        if "noaa_outcome_matches_result" in quality:
            clauses.append(f"NOAA outcomes agree for {quality['noaa_outcome_matches_result']:.2%}")
        if "missing_forecast_rows" in quality:
            clauses.append(f"{quality['missing_forecast_rows']:,} rows without a complete forecast day")
        if "incomplete_bracket_rows" in quality:
            clauses.append(f"{quality['incomplete_bracket_rows']:,} rows with an incomplete bracket set")
        if clauses:
            lines.append("Data quality: " + "; ".join(clauses) + ".")
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
    shared.line_chart(chart, "Mean predicted probability", "Observed frequency", "Model")
    with st.expander("Calibration table"):
        shared.table(
            pd.DataFrame(
                {
                    "Model": labels,
                    "Bin": [f"{lo:.1f}–{lo + 0.1:.1f}" for lo in reliability["bin_lo"]],
                    "Contracts": [int(n) for n in reliability["n"]],
                    "Mean predicted": [f"{x:.3f}" for x in reliability["mean_pred"]],
                    "Observed": [f"{x:.3f}" for x in reliability["obs_freq"]],
                }
            )
        )

    robustness = evidence["robustness"]
    st.subheader("By snapshot")
    lead = robustness[robustness["cut"] == "lead"].set_index("snapshot")
    lead = lead.reindex([s for s in inference.SNAPSHOT_LABELS if s in lead.index]).reset_index()
    shared.table(brier_table(lead, "snapshot", "Snapshot"))

    with st.expander("Robustness by liquidity and settlement source"):
        liquidity = robustness[robustness["cut"] == "liquidity"]
        rank = {label: i for i, label in enumerate(LIQUIDITY_ORDER)}
        liquidity = liquidity.sort_values("liquidity", key=lambda s: s.map(rank).fillna(len(rank)), kind="stable")
        liquidity = liquidity.assign(liquidity=liquidity["liquidity"].map(lambda b: LIQUIDITY_LABELS.get(b, b)))
        shared.table(brier_table(liquidity, "liquidity", "Liquidity"))
        sources = robustness[robustness["cut"] == "settle_source"]
        shared.table(brier_table(sources, "settle_source", "Settlement source"))
        twc = sources[sources["settle_source"] == "The Weather Company"]
        if not twc.empty:
            st.caption(
                f"The Weather Company settles {int(twc['events'].iloc[0])} of {test['events']} test city-days, "
                "so its numbers are less certain."
            )

    st.subheader("Limitations")
    st.markdown("\n".join(f"- {line}" for line in (FINDING_LIMITATIONS if finding is not None else LIMITATIONS)))

    st.subheader("Data sources")
    st.markdown(shared.data_sources(manifest), unsafe_allow_html=True)
