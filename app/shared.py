"""UI helpers shared by every page of the app.

Only presentation lives here; every number comes from weather_pricing.inference.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import streamlit as st

from weather_pricing import inference

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
}
FAMILY_LABELS = {
    "rule": "Rule-based",
    "weather": "Weather-only",
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


def comparison_sentence(change: float, baseline_label: str) -> str:
    """The Note Pricer's one-line comparison of the combined XGBoost model with the baseline.

    Below half a cent the change shows as $0.00 in the table, so the sentence says "the
    same" instead of "$0.00 less". Dollars are escaped for Streamlit Markdown.
    """
    start = "Under the combined XGBoost model the note is worth"
    if abs(change) < 0.005:
        return f"{start} the same as under the baseline ({baseline_label})."
    direction = "more" if change > 0 else "less"
    return f"{start} \\${abs(change):,.2f} {direction} than under the baseline ({baseline_label})."


ATTRIBUTION = """
---
**Data sources.** Market data: Kalshi public market data (contract terms, quotes and
outcomes). <a href="https://open-meteo.com/">Weather data by Open-Meteo.com</a>, licensed
under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>; the forecasts
were processed (hourly values chosen by run age and reduced to a daily maximum).
Observed temperatures: NOAA National Centers for Environmental Information,
Global Historical Climatology Network – Daily (GHCN-Daily), station USW00094728;
Menne, M.J., I. Durre, R.S. Vose, B.E. Gleason and T.G. Houston (2012): An overview of
the Global Historical Climatology Network-Daily database. *Journal of Atmospheric and
Oceanic Technology*, 29, 897–910. Data citation: Menne, M.J., I. Durre, B. Korzeniewski,
S. McNeill, K. Thomas, X. Yin, S. Anthony, R. Ray, R.S. Vose, B.E. Gleason and
T.G. Houston (2012): Global Historical Climatology Network - Daily (GHCN-Daily),
Version 3. NOAA National Climatic Data Center. doi:10.7289/V5D21VHZ.
NOAA provides the data with no warranty of any kind.
"""


def attribution() -> None:
    st.markdown(ATTRIBUTION, unsafe_allow_html=True)
