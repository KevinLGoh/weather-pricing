# Weather prediction markets as alternative data

This is a minimum viable research prototype. It tests one question:

> Do market-implied probabilities from weather prediction markets improve the estimation
> and pricing of weather-related risk beyond what meteorological data alone achieves?

The research case is the Kalshi markets on daily high temperature. New York (Central Park)
is the first city. Chicago (Midway) is the extension. Each contract pays if the official
daily high is in a bracket, for example "70–71°F" or "77° or above".

The prototype asks two things:

1. Do the market prices add information to an archived numerical weather prediction (NWP)
   forecast?
2. How much does the difference change the value of a hypothetical weather-linked note?

## What this is

A research pipeline and a small Streamlit app for a risk analyst who prices a
hypothetical temperature-linked note. The app answers one question: **how much does the
probability source (a weather model, the market, or both) change the value of the
note?** It is research only, not financial advice; the note is hypothetical and the
risk loading λ is illustrative.

## Features

Research pipeline (`src/weather_pricing/`):

- Collectors for Kalshi (contracts, outcomes, hourly candles), Open-Meteo (archived NWP
  runs by run age) and NOAA GHCN-Daily (observed highs), with a cached, throttled HTTP
  client.
- A point-in-time dataset: one row per contract and snapshot, built only from
  information published before the snapshot, with a data-quality report.
- Eight models: a rule-based Gaussian, weather-only (logistic, XGBoost), market-only
  (raw mid, renormalised mid, recalibrated logistic) and combined (logistic, XGBoost).
- Evaluation: Brier score, log loss, ECE, reliability tables, an event-clustered
  bootstrap, a forecast-encompassing regression and robustness cuts.
- A one-period note valuation with a Wang-transform risk loading.
- `evaluate --save-artifacts`: saves models, out-of-sample predictions, results and a
  manifest (commit, source-tree hash, package versions, data range, row counts, NOAA
  access date) to `app_artifacts/`.

App (`app/`, gets all its data through `src/weather_pricing/inference.py`):

- **Contract Explorer**: pick a city, test-period day, snapshot and contract; compare
  the probability of every model by family; see the market quote and the outcome.
- **Note Pricer**: value the note on that contract under every model; change the
  principal, loss fraction, coupon and λ; compare each model with a chosen baseline.
- **Model Evidence**: scores, bootstrap confidence intervals, the encompassing
  regression, calibration, results by snapshot, robustness, limitations and data
  sources.
- Every page: short instructions, loading and error messages without stack traces, a
  layout that works on phones, and the research-only notice.

## Installation

Python 3.12 (the version the app artifacts were built with). From the repo root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

If `python3.12` is not on the PATH, use the full path to a Python 3.12, for example
Homebrew's `/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv`. All later
commands assume the virtual environment is active.

`requirements.txt` pins the exact package versions of the artifacts and installs the
package itself (`.`), so run it from the repo root. For development, an editable install
works too: `pip install -e '.[dev,app]'`; it does not pin the versions the artifacts
were built with.

## Environment and API keys

No API keys are needed: Kalshi, Open-Meteo and NOAA are used through public, token-free
endpoints. The repo follows the `.env` pattern anyway: `.env.example` lists the
settings (none today); copy it to `.env` for local values. `.env` and `.env.*` are
ignored by git (only `.env.example` is tracked); never commit a filled-in `.env`.

## Quick start

```bash
pip install -e '.[dev,app]'
```

```bash
python -m weather_pricing.cli build --cities NY --start 2023-01-01
```

```bash
python -m weather_pricing.cli evaluate
```

```bash
pytest -q
```

`build` downloads all data from read-only public endpoints. It needs no API keys. It
caches each raw response under `data/raw/`. A rebuild still needs the network: the
Kalshi listings and historical cutoff, Open-Meteo chunks that end in the last 8 days and
NOAA's current year are always fetched again; candles and older weather data come from
the cache. `evaluate` and the app run offline. `build` writes
`data/processed/snapshots.parquet` and `results/data_quality.json`.

`evaluate` writes `results/summary.md` and the CSV and JSON detail files.

## Usage guide

All commands run from the repo root with the virtual environment active
(`source .venv/bin/activate`).

**1. Build the data and the results.**

```bash
python -m weather_pricing.cli build --cities NY --start 2023-01-01
python -m weather_pricing.cli evaluate
```

The first `build` for New York takes about 35–45 minutes, mostly Kalshi candles at
about 5 requests per second; it ends by printing the data-quality report. `evaluate`
takes seconds and prints the results summary (see Results).

**2. Build the app artifacts.**

```bash
python -m weather_pricing.cli evaluate --save-artifacts
```

This writes `app_artifacts/` (about 2 MB): `models.pkl`, `predictions.parquet`, the
results files and `manifest.json`. The app reads only this folder.

**3. Run the app.**

```bash
streamlit run app/streamlit_app.py
```

Start it from the repo root, so Streamlit finds `.streamlit/config.toml`, and open the
local URL it prints.

**4. Worked example.** In the Contract Explorer, pick New York, 2026-11-02, snapshot
D-1 14:00 (34 hours before the day ends) and the bracket 65–66°F:

- The market quote is a mid price of \$0.49 (bid \$0.48, ask \$0.51) with a spread of
  \$0.03.
- The probabilities are 20.0 % (rule-based Gaussian), 29.4 % (weather-only, XGBoost),
  48.1 % (market mid, renormalised) and 53.2 % (combined, XGBoost).
- The outcome: the daily high was 66°F, so the contract paid Yes.

Press **Price this contract**. With the default note (principal \$1,000, loss 50 % of
the principal, coupon \$0, λ = 0) the present value is \$852.87 under the weather-only
XGBoost model, \$759.56 under the renormalised market mid and \$733.95 under the
combined XGBoost model: the combined model values the note \$118.92 lower than the
weather-only baseline. With λ = 0.25 the three values are \$807.26, \$709.98 and
\$685.25. Model Evidence then shows how these models score on the whole test period.

**Screenshots** (in `docs/screenshots/`):

- Contract Explorer with the worked-example contract: selectors, quote, probabilities by
  family, outcome.

  ![Contract Explorer](docs/screenshots/01-contract-explorer.png)

- Note Pricer after "Price this contract": principal 1,000, loss 50 %, coupon 0,
  λ 0.25, baseline weather-only XGBoost.

  ![Note Pricer](docs/screenshots/02-note-pricer.png)

- Model Evidence: scores, bootstrap intervals, encompassing regression and conclusion.

  ![Model Evidence, scores](docs/screenshots/03-model-evidence-scores.png)

- Model Evidence: calibration chart and results by snapshot.

  ![Model Evidence, calibration](docs/screenshots/04-model-evidence-calibration.png)

- Note Pricer opened in a fresh session without a contract: the hint and the link back.

  ![Note Pricer without a contract](docs/screenshots/05-note-pricer-no-selection.png)

- Contract Explorer at phone width (about 390 px).

  ![Phone width](docs/screenshots/06-phone-width.png)

## Data

| Source | Endpoint | Use |
|---|---|---|
| Kalshi | `/markets`, `/historical/markets`, `.../candlesticks` (hourly) | Contract terms, outcomes, bid/ask/last, volume |
| Open-Meteo | previous-runs API, `temperature_2m_previous_day1..5` | Archived NWP forecasts at a known run age |
| NOAA NCEI | Access Data Service, GHCN-Daily `TMAX` | Independent check of outcomes, persistence feature |

### Results of the feasibility audit (plan Weeks 1–2)

- **The history is longer than we expected.** Kalshi has NYC daily-high markets from
  Aug 2021. The series name was `HIGHNY` until 24 Oct 2024, and `KXHIGHNY` after that
  date. For markets that settled before the exchange's historical cutoff, only the
  `/historical/*` endpoints return data.
- **The lead times are short.** A market opens at about 10:00 ET on the day before its
  target date. It stops trading at local midnight. Thus, the 7-day and 3-day horizons in
  the plan do not exist. The builder takes snapshots at D-1 14:00, D-1 22:00, D 08:00 and
  D 12:00 local time. These are 34 to 12 hours before the target day ends.
- **The settlement source changed.** Contracts for target days up to 2026-08-13 cite the
  NWS Daily Climate Report; from 2026-08-14 they cite The Weather Company. The rule-text
  extractor records the source for each contract; `data_quality.json` counts the
  contracts per source, and the robustness results (and the app's Model Evidence page)
  show the scores per source.
- **The contract structure changed.** In 2021–22, each event was a ladder of thresholds
  (">45°", ">47°"). From late 2022, the contracts in each event cover all temperatures: a
  low tail, 2°F brackets and a high tail. Thus, the quotes in one event can be
  renormalised so that they sum to one.
- **Some old listings have no structured strikes.** The builder reads their bracket from
  the rule text. This is the "LLM-assisted extraction" step in the plan, done here with a
  validated regex extractor. The builder sets the flag `strikes_from_text` on these rows.
- **Kalshi limits the request rate.** Above approximately 5 requests per second, Kalshi
  returns HTTP 429. Thus, the first download of one city-year of candles takes
  approximately 10 minutes.

### Prevention of look-ahead bias

Each row is one contract at one snapshot time. All features on a row were public at that
time:

- **Quotes:** the builder uses the last hourly candle that *closed* before the snapshot.
- **Forecasts:** for each valid hour `t`, the builder selects `previous_dayN` with
  `N ≥ ceil((t − snapshot + 6h) / 24h)`. This is the newest of Open-Meteo's
  `previous_day1..5` runs that had been published before the snapshot, assuming a run is
  published 6 hours after it starts. No run younger than `previous_day1` (started about
  a day before the valid hour) is available, so the forecast used is often 7–23 h old at
  the snapshot and is the same at D 08:00 and D 12:00 (see Limitations). The daily
  maximum comes from these hourly values.
- **Persistence:** the builder uses the observed high of the last local day that ended
  before the snapshot.
- **Splits:** the split is chronological by target date. All contracts and snapshots of a
  city-day have one outcome. Thus, they are always in the same split, and the bootstrap
  resamples whole city-days.

## Models

| Name | Family | Inputs |
|---|---|---|
| `R_gaussian_rule` | Rule-based | Forecast plus a training-period bias, with N(0, σ) error by city and lead bucket, integrated over the bracket |
| `A_logit`, `A_xgb` | Weather-only (Model A) | Gaussian bracket probability, bracket z-scores, lead, forecast revision, persistence anomaly, season |
| `B_market_mid`, `B_market_norm` | Market-only (Model B) | Raw mid, and mid renormalised within the event |
| `B_logit` | Market-only, recalibrated | Logit mid, spread, volume, staleness |
| `C_logit`, `C_xgb` | Combined (Model C) | A + B features |

We compare A and C with the same learner: logistic against logistic, and XGBoost against
XGBoost. Thus, any gain comes from the market data, not from the model type.

XGBoost selects its number of trees on the last 20% of the training period. Then it fits
again on the full training set.

The MVP does not include deep learning (LSTM/TFT) or an LLM extractor (Model D). The plan
recommends a narrow first prototype.

## Evaluation

- Brier score, log loss, expected calibration error (ECE) and reliability tables.
- **Event-clustered bootstrap** of the Brier differences, with a 95% CI and a two-sided
  p-value.
- **Forecast-encompassing regression** with cluster-robust standard errors:
  `logit P(y) = a + b1·logit(p_weather) + b2·logit(p_market)`. If `b2` is significant, the
  market has information that the weather model does not have. This test examines the
  central hypothesis of the plan more directly than accuracy near settlement does.
- Robustness cuts by snapshot time, liquidity (spread or staleness) and city.

## Valuation

Each contract has a hypothetical one-period note. The principal is N = $1,000. The note
loses L = 50% of the principal if the bracket event occurs:

    V_t = D(t,T) · (C + N · (1 − q_t · L)),   q_t = Φ(Φ⁻¹(p_t) + λ)

When λ = 0, V_t is the discounted expected payoff. This is the formula in Section 6 of the
plan. When λ > 0, the formula applies a Wang-transform risk loading. The loading shows a
point from the plan: forecast probabilities are not risk-adjusted pricing probabilities.

The outputs are:

- the change in value when Model C replaces Model A
- the RMSE of the value against the realised payoff (for reference only: with λ = 0 it
  equals D·N·L·√Brier, a rescaled Brier score, so it is not separate evidence)
- the value for a range of λ values

The horizon is approximately one day, so the discount factor is almost 1. Thus, the value
differences are approximately N·L·Δp.

## Results

These results are from the reproduced run of 2026-11-04, with New York data only (see
"Reproduction" below for the differences from the first run). The command was
`python -m weather_pricing.cli evaluate` with default settings. `results/summary.md` has
the full tables.

**Sample.** The sample has 1,401 city-days and 8,403 contracts, in 33,473 snapshot rows
(2023-01-01 to 2026-11-02). 32,951 rows are used; the other 522 lack a complete day of
forecast hours. The data quality is good:

- The bracket logic gives the same result as Kalshi for 100% of contracts.
- NOAA TMAX gives the same outcome for 99.98% of contracts.
- The rule text and the strikes agree for all contracts.
- 40 rows (0.1%) belong to an event whose brackets were not all quoted at that snapshot;
  they keep the raw mid instead of the renormalised one.
- In `data_quality.json`, `event_mid_sum_median` (1.07) is now computed over complete
  bracket sets only.

**Split.** The training set has 965 city-days (2023-01-01 to 2025-09-14), in 23,033 rows.
The test set has 414 city-days (2025-09-15 to 2026-11-02), in 9,918 rows.

| Model | Brier | Log loss | ECE |
|---|---|---|---|
| Rule-based Gaussian | 0.1437 | 0.462 | 0.054 |
| A: weather-only, logistic | 0.1345 | 0.434 | 0.015 |
| A: weather-only, XGBoost | 0.1278 | 0.412 | 0.018 |
| B: market mid, renormalised | **0.0941** | **0.293** | 0.015 |
| C: combined, logistic | 0.0954 | 0.296 | 0.022 |
| C: combined, XGBoost | 0.0950 | **0.293** | **0.013** |

1. **Market prices add much information to this weather baseline.** With XGBoost, the change
   from A to C decreases the Brier score by 0.033 (26%). The event-clustered 95% CI is
   0.028 to 0.037, and p < 0.001. The gain occurs at all snapshot times. At D-1 14:00,
   approximately 34 hours before the day ends, the market score is 0.109. The weather
   model score is 0.132.
2. **This weather baseline adds no information to market prices.** For C_xgb against B, the
   Brier difference is −0.0009 (CI −0.0024 to 0.0005, p = 0.20): C_xgb is not better than
   the market. C_logit is *worse* than the market (−0.0013, CI −0.0021 to −0.0005).
3. **The market prices contain the information in this forecast.** In the encompassing
   regression, the market coefficient is 1.22 (SE 0.06). The weather coefficient is −0.14
   (p = 0.12). A market coefficient above 1 shows that the quotes are slightly too close
   to 50%. But the recalibrated `B_logit` model did not get an out-of-sample gain from
   this.
4. **The result is robust.** It is true for all liquidity buckets and for both settlement
   sources. The gain is larger with The Weather Company (0.050) than with NWS (0.029). But
   The Weather Company settles only 81 of the 414 test city-days (2026-08-14 to
   2026-11-02).
5. **Valuation.** When the note uses Model C instead of Model A, its value changes by \$63
   on average. The 90th percentile is \$143 per \$1,000. The RMSE against the realised
   payoff falls from \$179 to \$154, but this is the Brier result in dollars
   (D·N·L·√Brier), not separate evidence. A Wang loading of λ = 0.25 decreases the mean
   value by approximately \$23. This does not show that the risk premium matters as much
   as the forecast source: λ = 0.25 is illustrative, not calibrated, and the \$23 is a
   shift of the mean value while the \$63 is the average size of contract-by-contract
   changes, so the two numbers measure different things.

**Interpretation.** The weather baseline is one deterministic NWP run (Open-Meteo best
match) from a public archive, with simple statistical post-processing. Traders very
probably use ensembles, MOS/NBM guidance and live station observations. At the D 08:00
and D 12:00 snapshots, the morning observations are already available. Thus, part of the
market advantage comes from nowcasting.

Thus, the data supports this claim: *Kalshi prices contain all the information in a
public deterministic NWP forecast for this type of event.* These results are about this
weather baseline only; they are not evidence that prediction markets are better than
meteorology in general. The next test is a stronger
weather model with ensemble spread, NBM and intraday observations up to the snapshot.

### Reproduction (2026-11-04)

The data was downloaded again and the evaluation rerun with default settings
(`build --cities NY --start 2023-01-01`, then `evaluate`). All conclusions of the first
run hold. The differences:

- **More data.** The data now ends on 2026-11-02, so there are 3 more city-days
  (1,401 instead of 1,398), and the train/test cut moved slightly (965/414 instead of
  963/413 city-days).
- **Stricter forecast rule.** A row now needs every forecast hour of the local day,
  including 23- and 25-hour days at the DST changes (before: at least 20 hours).
- **Bracket check.** Quotes are renormalised only when every bracket of the event is
  quoted at that snapshot. This affects 40 rows.
- **Scores** move by at most 0.0007 (Brier) and 0.005 (ECE).
- **Weather added to the market** (finding 2): the CI is wider than in the first run,
  and C_xgb is now slightly, not significantly, worse than the market. C_logit is
  significantly worse.
- **Renormalising quotes** makes no difference: the raw mid (`B_market_mid`, Brier
  0.0940) and the renormalised mid score the same (p = 0.56).
- The separate effect of the two rule changes was not measured: that would need a build
  with the old code.

## Limitations

- **Forecast age.** The builder cannot use a run younger than `previous_day1`, so at the
  snapshot the forecast is often 7–23 h old, and it is the same at D 08:00 and D 12:00.
  This likely overstates the market's advantage. A forecast source with intraday runs is
  future work.
- **Short horizons.** These markets have horizons of 34 hours or less. The results give no
  information about weather risk over many days or seasons, or about cat-bond horizons.
- **Approximate NWP availability.** The model assumes that a run becomes available 6
  hours after initialisation. The Open-Meteo "best match" model also joins several NWP
  models, and it has some gaps (for example, early Jan 2024). If a row has a missing
  forecast hour, the experiment removes that row from all models.
- **Stale markets.** The quotes are hourly candle closes. In illiquid hours, the previous
  quote continues. The builder sets the flag `stale` on these rows, and the evaluation
  analyses them separately.
- **Hypothetical instrument.** The note is not a traded security. The experiment uses no
  real cat-bond prices. The λ values are examples, not calibrated values.
- **Settled temperatures before 2025.** Kalshi does not give most of these values. NOAA
  TMAX gives the missing values. The experiment uses these values only to fit the
  Gaussian error model. The labels always come from the Kalshi result.
- **Weak weather baseline.** The weather model has no ensemble spread and no intraday
  observations (see Results).
- **One test window.** The experiment uses one chronological train/test split. The next
  robustness step is a rolling-origin evaluation.

## Known issues

- The app covers the test period only (2025-09-15 to 2026-11-02 in the current
  artifacts), because only those predictions are out-of-sample.
- A rebuild of the data needs the network (see Quick start); only `evaluate` and the app
  run offline.
- On Streamlit Community Cloud the Python version is chosen in the app's settings, not
  by this repo; it must be set to 3.12 by hand.
- In the Contract Explorer the contracts table shows the raw mid (for example 0.495),
  while the quote line rounds it to two decimals (\$0.49).
- Model Evidence shows liquidity buckets with their stored labels (`spread<=2c`, `2-5c`,
  `>5c`, `stale/no quote`).
- A `data_quality.json` written by an older `build` lacks the newer keys (for example
  `incomplete_bracket_rows`); run `build` again before `evaluate --save-artifacts`.

## Future enhancements

- A forecast source with intraday runs (removes the forecast-age limitation).
- Ensemble spread, NBM guidance and intraday observations for the weather baseline.
- Rolling-origin evaluation and event-level scores such as the ranked probability score.
- More cities (Chicago first), and live scoring of open Kalshi markets.
- Deep-learning models and an LLM-based rule extractor (Model D).
- Format the contracts table as dollar prices like the quote line.

## Dependencies

Pinned in `requirements.txt` (the versions the app artifacts were built with):

| Package | Version | Used for |
|---|---|---|
| numpy | 2.5.4 | arrays and numerics |
| pandas | 3.1.0 | the dataset, results and app tables |
| pyarrow | 26.0.1 | parquet files (snapshots, predictions) |
| requests | 2.34.2 | the HTTP client of the collectors |
| scikit-learn | 1.9.1 | logistic models and scaling |
| scipy | 1.18.1 | normal distribution, logit, statistics |
| xgboost | 3.4.1 | the boosted-tree models |
| streamlit | 1.66.1 | the app (optional `app` extra in `pyproject.toml`) |

`requirements-dev.txt` adds pytest 9.2.0 for the test suite. Everything else is the
Python standard library.

## Data sources, attribution and licences

- **Kalshi**: contract terms, quotes and outcomes from Kalshi's public market-data API;
  Kalshi's terms of use apply to that data.
- **Open-Meteo**: <a href="https://open-meteo.com/">Weather data by Open-Meteo.com</a>,
  licensed under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.
  The data was processed: hourly values were chosen by run age and reduced to a daily
  maximum.
- **NOAA NCEI GHCN-Daily**: Menne, M.J., I. Durre, B. Korzeniewski, S. McNeill,
  K. Thomas, X. Yin, S. Anthony, R. Ray, R.S. Vose, B.E. Gleason and T.G. Houston
  (2012): Global Historical Climatology Network - Daily (GHCN-Daily), Version 3. Subset
  used: station USW00094728, New York (Central Park), elements TMAX and TMIN. NOAA
  National Climatic Data Center. doi:10.7289/V5D21VHZ. Accessed 2026-11-03 for the
  current artifacts (the app shows the date recorded in `app_artifacts/manifest.json`).
  Paper: Menne, M.J., I. Durre, R.S. Vose, B.E. Gleason and T.G. Houston (2012): An
  overview of the Global Historical Climatology Network-Daily database. *Journal of
  Atmospheric and Oceanic Technology*, 29, 897–910. NOAA provides the data with no
  warranty of any kind.
- **Code**: the repository contains no licence file.

## Performance

Two caches keep the slow parts fast:

- **Cached, throttled data fetching** (`http.py`): every response is stored once under
  `data/raw/` (written atomically, so an interrupted download never leaves a broken
  file), requests to each source are spaced (Kalshi candles 0.2 s, Open-Meteo and NOAA
  0.5 s) and retried with back-off on HTTP 429/5xx, and candles download in 3 parallel
  threads. A rebuild only fetches what can change.
- **Cached artifact loading in the app** (`inference.py`): the predictions and the
  manifest are read once per server process (`functools.lru_cache`); the results files
  are read only when Model Evidence opens. The app never retrains or calls an API.

## Layout

```
src/weather_pricing/
  config.py          cities, snapshot times, availability assumptions
  http.py            cached, throttled GET
  collect/           kalshi.py, openmeteo.py, noaa.py
  rules.py           rule-text extraction and validation, bracket logic
  dataset.py         point-in-time snapshot builder and data-quality report
  models.py          rule / A / B / C models
  evaluate.py        scores, calibration, clustered bootstrap, encompassing test
  valuation.py       one-period note, Wang transform
  experiment.py      split, fit, score, value, write results/
  artifacts.py       save models, predictions, results and the manifest for the app
  inference.py       the app's data layer: contracts, predictions, note prices, evidence
  cli.py             build / evaluate (--save-artifacts)
app/                 Streamlit app: streamlit_app.py, the three pages, shared.py
app_artifacts/       what the app loads (built by evaluate --save-artifacts)
docs/screenshots/    README screenshots
tests/               offline tests (synthetic data, no network)
requirements*.txt    pinned versions for the app and the tests
```
