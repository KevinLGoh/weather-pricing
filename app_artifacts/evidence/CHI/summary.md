# Results summary

Train: 965 city-days (2023-01-01 to 2025-09-14), 23092 rows.
Test: 414 city-days (2025-09-15 to 2026-11-02), 9918 rows. Base rate 0.167.
Rows used 33010 of 33533 (dropped rows lack a forecast, quote or settlement value).

## Out-of-sample scores (lower is better)

| model | brier | log_loss | ece |
|---|---|---|---|
| R_gaussian_rule | 0.1413 | 0.4520 | 0.0411 |
| A_logit | 0.1348 | 0.4377 | 0.0267 |
| A_xgb | 0.1331 | 0.4306 | 0.0258 |
| B_market_mid | 0.1011 | 0.3154 | 0.0142 |
| B_market_norm | 0.1012 | 0.3154 | 0.0156 |
| B_logit | 0.1021 | 0.3164 | 0.0163 |
| C_logit | 0.1020 | 0.3163 | 0.0125 |
| C_xgb | 0.1013 | 0.3158 | 0.0080 |
| W_rule | 0.1189 | 0.3723 | 0.0089 |
| W_logit | 0.1192 | 0.3736 | 0.0076 |
| W_xgb | 0.1183 | 0.3714 | 0.0078 |
| X_logit | 0.1171 | 0.3656 | 0.0111 |
| X_xgb | 0.1160 | 0.3628 | 0.0079 |
| CW_xgb | 0.1017 | 0.3159 | 0.0112 |
| CX_xgb | 0.1016 | 0.3149 | 0.0125 |

## Event-clustered bootstrap: Brier(baseline) - Brier(challenger)

Positive means the challenger is better. 95% CI from resampling whole city-days.

| question | baseline | challenger | diff | ci_lo | ci_hi | p_value | n_events |
|---|---|---|---|---|---|---|---|
| Market data added to weather (logistic) | A_logit | C_logit | 0.03278 | 0.02801 | 0.03759 | 0.00000 | 414 |
| Market data added to weather (XGBoost) | A_xgb | C_xgb | 0.03175 | 0.02732 | 0.03614 | 0.00000 | 414 |
| Weather data added to market (logistic) | B_market_norm | C_logit | -0.00088 | -0.00187 | 0.00009 | 0.07700 | 414 |
| Weather data added to market (XGBoost) | B_market_norm | C_xgb | -0.00017 | -0.00085 | 0.00050 | 0.64400 | 414 |
| Market alone vs best weather-only | A_xgb | B_market_norm | 0.03192 | 0.02762 | 0.03623 | 0.00000 | 414 |
| Renormalising quotes within an event | B_market_mid | B_market_norm | -0.00002 | -0.00014 | 0.00010 | 0.74100 | 414 |
| Stronger weather vs Open-Meteo weather | A_xgb | W_xgb | 0.01479 | 0.01202 | 0.01763 | 0.00000 | 414 |
| Extra forecast sources in the weather model | W_xgb | X_xgb | 0.00224 | 0.00122 | 0.00323 | 0.00000 | 414 |
| Market vs stronger weather | W_xgb | B_market_norm | 0.01713 | 0.01389 | 0.02032 | 0.00000 | 414 |
| Market vs extended weather | X_xgb | B_market_norm | 0.01489 | 0.01157 | 0.01817 | 0.00000 | 414 |
| Stronger weather added to market | B_market_norm | CW_xgb | -0.00054 | -0.00150 | 0.00045 | 0.28000 | 414 |
| Extended weather added to market | B_market_norm | CX_xgb | -0.00041 | -0.00162 | 0.00080 | 0.52700 | 414 |

## Forecast-encompassing regression (test set, clustered by city-day)

logit P(yes) = a + b1 * logit(p_weather [A_logit]) + b2 * logit(p_market [B_market_norm])

| term | coef | se | z | p_value | ci_lo | ci_hi |
|---|---|---|---|---|---|---|
| const | 0.2673 | 0.0967 | 2.7652 | 0.0057 | 0.0778 | 0.4568 |
| logit_weather | 0.0633 | 0.0625 | 1.0116 | 0.3117 | -0.0593 | 0.1858 |
| logit_market | 1.1667 | 0.0569 | 20.5057 | 0.0000 | 1.0552 | 1.2783 |

## Brier by snapshot time

| snapshot | n | events | A_xgb | B_market_norm | C_logit | C_xgb | W_xgb | X_xgb | A_minus_C_xgb |
|---|---|---|---|---|---|---|---|---|---|
| D-0 08:00 | 2484 | 414 | 0.1312 | 0.0955 | 0.0969 | 0.0956 | 0.1172 | 0.1143 | 0.0357 |
| D-0 12:00 | 2484 | 414 | 0.1312 | 0.0825 | 0.0843 | 0.0834 | 0.1054 | 0.1042 | 0.0479 |
| D-1 14:00 | 2472 | 412 | 0.1366 | 0.1171 | 0.1175 | 0.1174 | 0.1268 | 0.1248 | 0.0192 |
| D-1 22:00 | 2478 | 413 | 0.1332 | 0.1096 | 0.1096 | 0.1090 | 0.1237 | 0.1210 | 0.0242 |

## Brier by liquidity

| liquidity | n | events | A_xgb | B_market_norm | C_logit | C_xgb | W_xgb | X_xgb | A_minus_C_xgb |
|---|---|---|---|---|---|---|---|---|---|
| 2-5c | 2174 | 403 | 0.1868 | 0.1484 | 0.1500 | 0.1489 | 0.1715 | 0.1696 | 0.0379 |
| >5c | 597 | 192 | 0.1972 | 0.1523 | 0.1528 | 0.1540 | 0.1775 | 0.1732 | 0.0432 |
| spread<=2c | 6865 | 414 | 0.1141 | 0.0845 | 0.0852 | 0.0844 | 0.0996 | 0.0975 | 0.0297 |
| stale/no quote | 282 | 104 | 0.0441 | 0.0348 | 0.0354 | 0.0348 | 0.0366 | 0.0353 | 0.0092 |

## Brier by city

| city | n | events | A_xgb | B_market_norm | C_logit | C_xgb | W_xgb | X_xgb | A_minus_C_xgb |
|---|---|---|---|---|---|---|---|---|---|
| CHI | 9918 | 414 | 0.1331 | 0.1012 | 0.1020 | 0.1013 | 0.1183 | 0.1160 | 0.0317 |

## Brier by settlement source

| settle_source | n | events | A_xgb | B_market_norm | C_logit | C_xgb | W_xgb | X_xgb | A_minus_C_xgb |
|---|---|---|---|---|---|---|---|---|---|
| NWS | 7974 | 333 | 0.1345 | 0.1023 | 0.1032 | 0.1027 | 0.1197 | 0.1178 | 0.0317 |
| The Weather Company | 1944 | 81 | 0.1274 | 0.0965 | 0.0973 | 0.0956 | 0.1124 | 0.1089 | 0.0319 |

## Valuation of the hypothetical note (N=$1,000, L=50%)

- Mean |V(C_xgb) - V(A_xgb)|: $66.23 (90th pct $130.64); share of rows moving > $5: 92.4%
- RMSE of model value vs realised payoff: A_xgb $182.37, B_market_norm $159.00, C_xgb $159.13
- Mean value C_xgb by Wang lambda: 0.0: $914.72, 0.1: $905.49, 0.25: $890.60, 0.5: $863.22
