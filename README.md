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

1. **Market prices add much information to the weather model.** With XGBoost, the change
   from A to C decreases the Brier score by 0.033 (26%). The event-clustered 95% CI is
   0.028 to 0.037, and p < 0.001. The gain occurs at all snapshot times. At D-1 14:00,
   approximately 34 hours before the day ends, the market score is 0.109. The weather
   model score is 0.132.
2. **The weather model adds no information to market prices.** For C_xgb against B, the
   Brier difference is −0.0009 (CI −0.0024 to 0.0005, p = 0.20): C_xgb is not better than
   the market. C_logit is *worse* than the market (−0.0013, CI −0.0021 to −0.0005).
3. **The market prices contain the information in the forecast.** In the encompassing
   regression, the market coefficient is 1.22 (SE 0.06). The weather coefficient is −0.14
   (p = 0.12). A market coefficient above 1 shows that the quotes are slightly too close
   to 50%. But the recalibrated `B_logit` model did not get an out-of-sample gain from
   this.
4. **The result is robust.** It is true for all liquidity buckets and for both settlement
   sources. The gain is larger with The Weather Company (0.050) than with NWS (0.029). But
   the test set has only 81 days with The Weather Company as the source.
5. **Valuation.** When the note uses Model C instead of Model A, its value changes by $63
   on average. The 90th percentile is $143 per $1,000. The RMSE against the realised
   payoff decreases from $179 to $154. A Wang loading of λ = 0.25 decreases the mean value
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
