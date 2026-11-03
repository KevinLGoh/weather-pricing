"""Entry point of the research app.

Run locally from the repo root (Streamlit reads .streamlit/config.toml there):

    streamlit run app/streamlit_app.py
"""

import streamlit as st

import shared

st.set_page_config(page_title="Weather note pricing (research)", layout="centered")
shared.init()
st.sidebar.warning(shared.NOTICE)

page = st.navigation(
    [
        st.Page("contract_explorer.py", title="Contract Explorer", default=True),
        st.Page("note_pricer.py", title="Note Pricer"),
    ]
)
# Reserve the page area and draw the footer first: a page that stops on an error
# (shared.guarded calls st.stop) must still show the data attribution.
body = st.container()
shared.attribution()
with body:
    page.run()
