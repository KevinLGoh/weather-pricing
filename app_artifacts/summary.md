# Results summary

Train: 965 city-days (2023-01-01 to 2025-09-14), 23033 rows.
Test: 414 city-days (2025-09-15 to 2026-11-02), 9918 rows. Base rate 0.167.
Rows used 32951 of 33473 (dropped rows lack a forecast, quote or settlement value).

## Out-of-sample scores (lower is better)

| model | brier | log_loss | ece |
|---|---|---|---|
| R_gaussian_rule | 0.1437 | 0.4615 | 0.0541 |
| A_logit | 0.1345 | 0.4335 | 0.0154 |
| A_xgb | 0.1278 | 0.4116 | 0.0176 |
| B_market_mid | 0.0940 | 0.2928 | 0.0158 |
| B_market_norm | 0.0941 | 0.2928 | 0.0153 |
| B_logit | 0.0952 | 0.2947 | 0.0249 |
| C_logit | 0.0954 | 0.2957 | 0.0222 |
| C_xgb | 0.0950 | 0.2928 | 0.0130 |

## Event-clustered bootstrap: Brier(baseline) - Brier(challenger)

Positive means the challenger is better. 95% CI from resampling whole city-days.

| question | baseline | challenger | diff | ci_lo | ci_hi | p_value | n_events |
|---|---|---|---|---|---|---|---|
| Market data added to weather (logistic) | A_logit | C_logit | 0.03915 | 0.03429 | 0.04415 | 0.00000 | 414 |
| Market data added to weather (XGBoost) | A_xgb | C_xgb | 0.03278 | 0.02805 | 0.03740 | 0.00000 | 414 |
| Weather data added to market (logistic) | B_market_norm | C_logit | -0.00131 | -0.00212 | -0.00048 | 0.00000 | 414 |
| Weather data added to market (XGBoost) | B_market_norm | C_xgb | -0.00094 | -0.00237 | 0.00053 | 0.20000 | 414 |
| Market alone vs best weather-only | A_xgb | B_market_norm | 0.03371 | 0.02905 | 0.03837 | 0.00000 | 414 |
| Renormalising quotes within an event | B_market_mid | B_market_norm | -0.00005 | -0.00020 | 0.00011 | 0.56200 | 414 |

## Forecast-encompassing regression (test set, clustered by city-day)

logit P(yes) = a + b1 * logit(p_weather [A_logit]) + b2 * logit(p_market [B_market_norm])

| term | coef | se | z | p_value | ci_lo | ci_hi |
|---|---|---|---|---|---|---|
| const | -0.0019 | 0.1282 | -0.0151 | 0.9880 | -0.2532 | 0.2493 |
| logit_weather | -0.1371 | 0.0888 | -1.5443 | 0.1225 | -0.3112 | 0.0369 |
| logit_market | 1.2195 | 0.0598 | 20.4080 | 0.0000 | 1.1024 | 1.3366 |

## Brier by snapshot time

| snapshot | n | events | A_xgb | B_market_norm | C_logit | C_xgb | A_minus_C_xgb |
|---|---|---|---|---|---|---|---|
| D-0 08:00 | 2484 | 414 | 0.1250 | 0.0883 | 0.0901 | 0.0894 | 0.0356 |
| D-0 12:00 | 2484 | 414 | 0.1250 | 0.0764 | 0.0785 | 0.0776 | 0.0473 |
| D-1 14:00 | 2472 | 412 | 0.1315 | 0.1094 | 0.1096 | 0.1099 | 0.0216 |
| D-1 22:00 | 2478 | 413 | 0.1297 | 0.1023 | 0.1035 | 0.1031 | 0.0265 |

## Brier by liquidity

| liquidity | n | events | A_xgb | B_market_norm | C_logit | C_xgb | A_minus_C_xgb |
|---|---|---|---|---|---|---|---|
| 2-5c | 2174 | 409 | 0.1956 | 0.1562 | 0.1584 | 0.1573 | 0.0383 |
| >5c | 413 | 180 | 0.2089 | 0.1680 | 0.1691 | 0.1700 | 0.0389 |
| spread<=2c | 7010 | 414 | 0.1070 | 0.0741 | 0.0752 | 0.0750 | 0.0320 |
| stale/no quote | 321 | 141 | 0.0181 | 0.0142 | 0.0144 | 0.0133 | 0.0048 |

## Brier by city

| city | n | events | A_xgb | B_market_norm | C_logit | C_xgb | A_minus_C_xgb |
|---|---|---|---|---|---|---|---|
| NY | 9918 | 414 | 0.1278 | 0.0941 | 0.0954 | 0.0950 | 0.0328 |

## Brier by settlement source

| settle_source | n | events | A_xgb | B_market_norm | C_logit | C_xgb | A_minus_C_xgb |
|---|---|---|---|---|---|---|---|
| NWS | 7974 | 333 | 0.1270 | 0.0968 | 0.0982 | 0.0983 | 0.0286 |
| The Weather Company | 1944 | 81 | 0.1312 | 0.0828 | 0.0837 | 0.0814 | 0.0498 |

## Valuation of the hypothetical note (N=$1,000, L=50%)

- Mean |V(C_xgb) - V(A_xgb)|: $63.33 (90th pct $142.92); share of rows moving > $5: 91.1%
- RMSE of model value vs realised payoff: A_xgb $178.72, B_market_norm $153.34, C_xgb $154.10
- Mean value C_xgb by Wang lambda: 0.0: $910.90, 0.1: $902.11, 0.25: $888.12, 0.5: $862.77
