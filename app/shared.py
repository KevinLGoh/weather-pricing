"""UI helpers shared by every page of the app.

Only presentation lives here; every number comes from weather_pricing.inference.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from weather_pricing import inference
from weather_pricing.config import CITIES

# The artifacts sit next to the app folder (repo root / app_artifacts), wherever the
# package itself is installed.
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "app_artifacts"

NOTICE = "Research only, not financial advice."
GENERIC_ERROR = (
    "Something went wrong while loading this page. Please reload it; "
    "if the problem stays, the app artifacts may need to be rebuilt."
)

MODEL_LABELS = {
    "R_gaussian_rule": "Rule-based Gaussian",
    "A_logit": "Weather-only, logistic",
    "A_xgb": "Weather-only, XGBoost",
    "B_market_mid": "Market mid (raw)",
    "B_market_norm": "Market mid, renormalised",
    "B_logit": "Market, recalibrated",
    "C_logit": "Combined, logistic",
    "C_xgb": "Combined, XGBoost",
    "W_rule": "Stronger weather, rule",
    "W_logit": "Stronger weather, logistic",
    "W_xgb": "Stronger weather, XGBoost",
    "X_logit": "Extended weather, logistic",
    "X_xgb": "Extended weather, XGBoost",
    "CW_xgb": "Stronger weather + market, XGBoost",
    "CX_xgb": "Extended weather + market, XGBoost",
}
FAMILY_LABELS = {
    "rule": "Rule-based",
    "weather": "Weather-only",
    "stronger_weather": "Stronger weather",
    "extended_weather": "Extended weather",
    "market": "Market",
    "combined": "Combined",
}

log = logging.getLogger(__name__)


def init() -> None:
    """Point the inference layer at the artifacts next to the app."""
    inference.set_artifacts_dir(ARTIFACTS_DIR)


def page_header(title: str, instructions: list[str]) -> None:
    st.title(title)
    st.warning(f"{NOTICE} The note is hypothetical.")
    st.markdown("\n".join(f"- {line}" for line in instructions))


@contextmanager
def guarded(message: str = "Loading…"):
    """Run page code with a spinner; show errors as messages, never as stack traces.

    Streamlit's own control flow (st.stop, st.rerun, st.switch_page) raises
    BaseException subclasses, which pass through untouched.
    """
    try:
        with st.spinner(message):
            yield
    except inference.InferenceError as e:
        st.error(str(e))
        st.stop()
    except Exception:
        log.exception("Unexpected error in the app")
        st.error(GENERIC_ERROR)
        st.stop()


TABLE_ROW_PX = 35  # Streamlit's dataframe row height


def table(frame: pd.DataFrame) -> None:
    """A table tall enough to show every row, because Streamlit otherwise clips after 10."""
    st.dataframe(frame, hide_index=True, height=(len(frame) + 1) * TABLE_ROW_PX + 3)


def metric_rows(models: list[str]) -> list[list[str]]:
    """Rows of at most three metrics: four in one row would cut their labels off."""
    if len(models) == 4:
        return [models[:2], models[2:]]
    return [models[i : i + 3] for i in range(0, len(models), 3)]


def bar_chart(labels: list[str], values: list[float], title: str, domain: list[float] | None = None) -> None:
    """A static horizontal bar chart with full labels.

    Not st.bar_chart: that cuts long labels off, and scrolling the page over it zooms the
    axis, which can show impossible values such as negative probabilities.
    """
    frame = pd.DataFrame({"Model": labels, "Value": values})
    scale = alt.Scale(domain=domain) if domain else alt.Undefined
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("Value:Q", title=title, scale=scale),
            y=alt.Y("Model:N", sort=None, title=None, axis=alt.Axis(labelLimit=0)),
            tooltip=["Model", alt.Tooltip("Value:Q", format=".3f", title=title)],
        )
        .properties(height=alt.Step(22))
    )
    st.altair_chart(chart, width="stretch")


def line_chart(frame: pd.DataFrame, x: str, y: str, color: str) -> None:
    """A static line chart on the unit square, with every legend entry readable."""
    unit = alt.Scale(domain=[0, 1])
    chart = (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{x}:Q", scale=unit),
            y=alt.Y(f"{y}:Q", scale=unit),
            color=alt.Color(f"{color}:N", legend=alt.Legend(orient="bottom", columns=2, labelLimit=0)),
        )
        .properties(height=380)
    )
    st.altair_chart(chart, width="stretch")


def city_label(key: str) -> str:
    """'New York (Central Park)' for a configured city, 'All cities' for the pooled rows."""
    if key == "ALL":
        return "All cities"
    return CITIES[key].name if key in CITIES else key


def comparison_sentence(change: float, baseline_label: str, model_label: str = "the combined XGBoost model") -> str:
    """The Note Pricer's one-line comparison of a combined model with the baseline.

    Below half a cent the change shows as $0.00 in the table, so the sentence says "the
    same" instead of "$0.00 less". Dollars are escaped for Streamlit Markdown.
    """
    start = f"Under {model_label} the note is worth"
    if abs(change) < 0.005:
        return f"{start} the same as under the baseline ({baseline_label})."
    direction = "more" if change > 0 else "less"
    return f"{start} \\${abs(change):,.2f} {direction} than under the baseline ({baseline_label})."


MARKET_AND_WEATHER = (
    "Market data: Kalshi public market data (contract terms, quotes and outcomes). "
    '<a href="https://open-meteo.com/">Weather data by Open-Meteo.com</a>, licensed under '
    '<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>; the forecasts were processed '
    "(hourly values chosen by run age and reduced to a daily maximum). NWS model guidance (GFS and NAM MOS, "
    "NBM, LAMP) and station observations from the Iowa Environmental Mesonet, Iowa State University."
)
GHCN_PAPER = (
    "Menne, M.J., I. Durre, R.S. Vose, B.E. Gleason and T.G. Houston (2012): An overview of the Global "
    "Historical Climatology Network-Daily database. *Journal of Atmospheric and Oceanic Technology*, 29, 897–910."
)
GHCN_DATASET = (
    "Menne, M.J., I. Durre, B. Korzeniewski, S. McNeill, K. Thomas, X. Yin, S. Anthony, R. Ray, R.S. Vose, "
    "B.E. Gleason and T.G. Houston (2012): Global Historical Climatology Network - Daily (GHCN-Daily), "
    "Version 3."
)


def data_sources(manifest: dict | None) -> str:
    """The data attribution, with the GHCN-Daily data citation in NOAA's recommended form.

    The subset (stations of the manifest's cities, from config.CITIES) and the access date
    (manifest "noaa_accessed") are only known with a manifest; without one they are left out.
    """
    citation = [GHCN_DATASET]
    if manifest is not None:
        for city in manifest.get("cities", []):
            if city in CITIES:
                c = CITIES[city]
                citation.append(f"Subset used: station {c.ghcnd}, {c.name}, elements TMAX and TMIN.")
            else:  # say so instead of failing (the footer must never fail) or hiding the city
                citation.append(f"Subset used: station not configured for {city}.")
    citation.append("NOAA National Climatic Data Center. doi:10.7289/V5D21VHZ.")
    if manifest is not None:
        accessed = manifest.get("noaa_accessed")
        citation.append(
            "Access date not recorded." if accessed in (None, "unknown") else f"Accessed {accessed}."
        )
    return (
        f"{MARKET_AND_WEATHER} Observed temperatures: NOAA National Centers for Environmental "
        f"Information, Global Historical Climatology Network – Daily (GHCN-Daily). Data citation: "
        f"{' '.join(citation)} Paper: {GHCN_PAPER} NOAA provides the data with no warranty of any kind."
    )


def attribution() -> None:
    """Footer on every page. It must never fail, also when the artifacts are missing."""
    try:
        manifest = inference.load_manifest()
    except inference.InferenceError as e:  # missing or unreadable artifacts: the page shows the error
        log.info("Footer shows the data sources without the manifest: %s", type(e).__name__)
        manifest = None
    except Exception:
        log.exception("Footer could not read the app manifest")
        manifest = None
    st.markdown(f"---\n**Data sources.** {data_sources(manifest)}", unsafe_allow_html=True)
