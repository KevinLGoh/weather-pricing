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

## Quick start

```bash
pip install -e '.[dev]'
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
caches each raw response under `data/raw/`, so later rebuilds run offline. It writes
`data/processed/snapshots.parquet` and `results/data_quality.json`.

`evaluate` writes `results/summary.md` and the CSV and JSON detail files.

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
- **The settlement source changed.** Rules before 2025 cite the NWS Daily Climate Report.
  Current rules cite "The Weather Company" for station `CLINYC`. The rule-text extractor
  records the source for each contract. `data_quality.json` shows how frequently each
  source agrees with NOAA.
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
  `N ≥ ceil((t − snapshot + 6h) / 24h)`. This is the newest run that was initialised
  *and* published before the snapshot. The formula assumes that a run becomes available
  6 hours after initialisation. The daily maximum comes from these hourly values.
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
- the RMSE of the value from each model against the realised payoff
- the value for a range of λ values

The horizon is approximately one day, so the discount factor is almost 1. Thus, the value
differences are approximately N·L·Δp.

## Results

These results are from the first run, with New York data only. The command was
`python -m weather_pricing.cli evaluate` with default settings. `results/summary.md` has
the full tables.

**Sample.** The sample has 1,398 city-days and 8,385 contracts, in 33,401 snapshot rows.
The data quality is good:

- The bracket logic gives the same result as Kalshi for 100% of contracts.
- NOAA TMAX gives the same outcome for 99.98% of contracts.
- The rule text and the strikes agree for all contracts.

**Split.** The training set has 963 city-days (2023-01 to 2025-09). The test set has 413
city-days (2025-09 to 2026-10), in 9,894 rows.

| Model | Brier | Log loss | ECE |
|---|---|---|---|
| Rule-based Gaussian | 0.1441 | 0.463 | 0.055 |
| A: weather-only, logistic | 0.1344 | 0.433 | 0.014 |
| A: weather-only, XGBoost | 0.1274 | 0.409 | 0.013 |
| B: market mid, renormalised | **0.0942** | 0.293 | 0.016 |
| C: combined, logistic | 0.0955 | 0.296 | 0.022 |
| C: combined, XGBoost | 0.0943 | **0.292** | **0.011** |

1. **Market prices add much information to the weather model.** With XGBoost, the change
   from A to C decreases the Brier score by 0.033 (26%). The event-clustered 95% CI is
   0.028 to 0.038, and p < 0.001. The gain occurs at all snapshot times. At D-1 14:00,
   approximately 34 hours before the day ends, the market score is 0.110. The weather
   model score is 0.131.
2. **The weather model adds no information to market prices.** For C_xgb against B, the
   Brier difference is −0.0001 (CI −0.001 to 0.001). C_logit is slightly *worse* than the
   raw market.
3. **The market prices contain the information in the forecast.** In the encompassing
   regression, the market coefficient is 1.22 (SE 0.06). The weather coefficient is −0.13
   (p = 0.15). A market coefficient above 1 shows that the quotes are slightly too close
   to 50%. But the recalibrated `B_logit` model did not get an out-of-sample gain from
   this.
4. **The result is robust.** It is true for all liquidity buckets and for both settlement
   sources. The gain is larger with The Weather Company (0.048) than with NWS (0.030). But
   the test set has only 78 days with The Weather Company as the source.
5. **Valuation.** When the note uses Model C instead of Model A, its value changes by $62
   on average. The 90th percentile is $138 per $1,000. The RMSE against the realised
   payoff decreases from $178 to $153. A Wang loading of λ = 0.25 decreases the mean value
   by approximately $23. This is the same order as the gain from better forecasts. Thus,
   the risk-premium assumption is approximately as important as the source of the
   probability.

**Interpretation.** The weather baseline is one deterministic NWP run (Open-Meteo best
match) from a public archive, with simple statistical post-processing. Traders very
probably use ensembles, MOS/NBM guidance and live station observations. At the D 08:00
and D 12:00 snapshots, the morning observations are already available. Thus, part of the
market advantage comes from nowcasting.

Thus, the data supports this claim: *Kalshi prices contain all the information in a
public deterministic NWP forecast for this type of event.* The data does not support the
claim that prediction markets are better than meteorology. The next test is a stronger
weather model with ensemble spread, NBM and intraday observations up to the snapshot.

## Limitations

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
  cli.py             build / evaluate
tests/               offline tests (synthetic data, no network)
```
