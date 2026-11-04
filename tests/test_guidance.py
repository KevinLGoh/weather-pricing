from datetime import date

import numpy as np
import pandas as pd

from weather_pricing import guidance
from weather_pricing.collect import iem
from weather_pricing.config import GUIDANCE_LAG_HOURS, OBS_PUBLISH_MINUTES

TZ = "America/New_York"
TARGET = date(2026, 7, 4)
# 12:00 EDT on the target day.
SNAP = pd.Timestamp("2026-07-04 16:00", tz="UTC")
HIGH_TIME = pd.Timestamp("2026-07-05 00:00", tz="UTC")


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


def _runs(frame):
    return guidance.Runs.from_frame(frame)


def _all_runs(gfs=None, nbs=None, lav=None):
    empty = pd.DataFrame(columns=iem.MOS_COLUMNS).astype({"runtime": "datetime64[ns, UTC]", "ftime": "datetime64[ns, UTC]"})
    return {
        "GFS": _runs(empty if gfs is None else gfs),
        "NBS": _runs(empty if nbs is None else nbs),
        "LAV": _runs(empty if lav is None else lav),
    }


def _row(runtime, ftime, n_x=np.nan, tmp=np.nan, txn=np.nan, xnd=np.nan):
    return {"runtime": _ts(runtime), "ftime": ftime, "n_x": n_x, "tmp": tmp, "txn": txn, "xnd": xnd}


def _no_obs():
    return pd.DataFrame({"valid": pd.to_datetime([], utc=True), "tmpf": []})


def test_climate_day_uses_standard_time_all_year():
    # EST is UTC-5 even in July, so the climate day starts at 05:00 UTC.
    assert guidance.climate_day_bounds(TARGET, TZ) == (_ts("2026-07-04 05:00"), _ts("2026-07-05 05:00"))
    assert guidance.climate_day_bounds(date(2026, 1, 15), TZ)[0] == _ts("2026-01-15 05:00")
    assert guidance.climate_day_bounds(TARGET, "America/Chicago")[0] == _ts("2026-07-04 06:00")


def test_unreleased_runs_are_never_used():
    # The 12Z GFS is released at 17Z under the lag, after the 16Z snapshot.
    assert GUIDANCE_LAG_HOURS["GFS"] > 4
    gfs = pd.DataFrame(
        [_row("2026-07-04 06:00", HIGH_TIME, n_x=80.0), _row("2026-07-04 12:00", HIGH_TIME, n_x=999.0)]
    )
    out = guidance.snapshot_features(SNAP, TARGET, TZ, _all_runs(gfs=gfs), _no_obs())
    assert out["gfs_max"] == 80.0
    assert out["gfs_age_h"] == 10.0


def test_a_midday_run_without_the_high_falls_back_to_the_previous_run():
    # A late-morning run no longer forecasts today's high, so its row has no value.
    gfs = pd.DataFrame(
        [_row("2026-07-04 06:00", HIGH_TIME, n_x=80.0), _row("2026-07-04 09:00", HIGH_TIME)]
    )
    out = guidance.snapshot_features(SNAP, TARGET, TZ, _all_runs(gfs=gfs), _no_obs())
    assert (out["gfs_max"], out["gfs_age_h"]) == (80.0, 10.0)


def test_the_routine_report_before_a_snapshot_is_public():
    obs = pd.DataFrame({"valid": [_ts("2026-07-04 15:51")], "tmpf": [81.0]})
    assert guidance.snapshot_features(SNAP, TARGET, TZ, _all_runs(), obs)["obs_latest"] == 81.0


def test_lag_scale_moves_the_release_boundary():
    gfs = pd.DataFrame([_row("2026-07-04 06:00", HIGH_TIME, n_x=80.0), _row("2026-07-04 00:00", HIGH_TIME, n_x=70.0)])
    lags = {k: v * 3 for k, v in GUIDANCE_LAG_HOURS.items()}  # 15 h: the 06Z run is out at 21Z
    out = guidance.snapshot_features(SNAP, TARGET, TZ, _all_runs(gfs=gfs), _no_obs(), lags)
    assert out["gfs_max"] == 70.0


def test_observations_stop_at_the_public_cutoff_and_start_with_the_climate_day():
    public = SNAP - pd.Timedelta(minutes=OBS_PUBLISH_MINUTES)
    obs = pd.DataFrame(
        {
            "valid": [
                _ts("2026-07-04 04:51"),  # previous climate day
                _ts("2026-07-04 05:51"),
                public,
                public + pd.Timedelta(minutes=1),  # not public yet
            ],
            "tmpf": [120.0, 70.0, 75.0, 130.0],
        }
    )
    out = guidance.snapshot_features(SNAP, TARGET, TZ, _all_runs(), obs)
    assert out["obs_max_sofar"] == 75.0
    assert out["obs_latest"] == 75.0
    assert out["obs_age_h"] == 0.0


def test_no_observations_before_the_climate_day_starts():
    obs = pd.DataFrame({"valid": [_ts("2026-07-03 18:00")], "tmpf": [90.0]})
    snap = pd.Timestamp("2026-07-03 18:00", tz="UTC")  # D-1 14:00 EDT
    assert np.isnan(guidance.snapshot_features(snap, TARGET, TZ, _all_runs(), obs)["obs_max_sofar"])


def test_nbm_gives_the_high_and_its_spread():
    nbs = pd.DataFrame(
        [
            _row("2026-07-04 13:00", HIGH_TIME, txn=88.0, xnd=2.5),
            _row("2026-07-04 13:00", _ts("2026-07-04 15:00"), tmp=80.0),
        ]
    )
    obs = pd.DataFrame({"valid": [_ts("2026-07-04 15:40")], "tmpf": [83.0]})
    out = guidance.snapshot_features(SNAP, TARGET, TZ, _all_runs(nbs=nbs), obs)
    assert (out["nbm_max"], out["nbm_spread"], out["nbm_age_h"]) == (88.0, 2.5, 3.0)
    assert out["obs_minus_nbm"] == 3.0


def test_lamp_needs_to_cover_the_rest_of_the_warm_hours():
    # Run at 15Z, usable at 16Z. Hours inside the climate day after the snapshot count.
    hours = pd.date_range("2026-07-04 16:00", "2026-07-05 02:00", freq="h", tz="UTC")
    lav = pd.DataFrame([_row("2026-07-04 15:00", t, tmp=70.0 + i) for i, t in enumerate(hours)])
    out = guidance.snapshot_features(SNAP, TARGET, TZ, _all_runs(lav=lav), _no_obs())
    assert out["lamp_max"] == 80.0

    short = lav[lav["ftime"] < _ts("2026-07-04 20:00")]
    assert np.isnan(guidance.snapshot_features(SNAP, TARGET, TZ, _all_runs(lav=short), _no_obs())["lamp_max"])


def test_old_runs_are_ignored():
    gfs = pd.DataFrame([_row("2026-06-30 00:00", HIGH_TIME, n_x=80.0)])
    assert np.isnan(guidance.snapshot_features(SNAP, TARGET, TZ, _all_runs(gfs=gfs), _no_obs())["gfs_max"])


def test_enrich_merges_one_lookup_per_city_day_snapshot(monkeypatch):
    gfs = pd.DataFrame([_row("2026-07-04 06:00", HIGH_TIME, n_x=80.0)])
    monkeypatch.setattr(iem, "station_temps", lambda *a: _no_obs())
    monkeypatch.setattr(
        iem, "guidance", lambda icao, model, *a: gfs if model == "GFS" else gfs.iloc[0:0]
    )
    df = pd.DataFrame(
        {"city": "NY", "target_date": [TARGET, TARGET], "snapshot_ts": [SNAP, SNAP], "ticker": ["a", "b"]}
    )
    out = guidance.enrich(df)
    assert len(out) == 2
    assert out["gfs_max"].tolist() == [80.0, 80.0]
    assert set(guidance.FEATURES) <= set(out.columns)


def test_parse_archive_csv_and_reject_error_pages():
    asos = iem.parse_asos("station,valid,tmpf\nNYC,2024-07-01 00:04,73.00\nNYC,2024-07-01 00:41,\n")
    assert asos["tmpf"].tolist() == [73.0]
    assert asos["valid"].iloc[0] == _ts("2024-07-01 00:04")
    mos = iem.parse_mos("runtime,ftime,model,n_x,tmp\n2024-07-01 00:00:00,2024-07-02 00:00:00,GFS,80.0,75\n")
    assert mos["n_x"].iloc[0] == 80.0
    assert np.isnan(mos["txn"].iloc[0])  # GFS rows have no NBM columns
    try:
        iem.parse_mos("Too many requests from your IP address, slow down.")
    except ValueError as exc:
        assert "unexpected" in str(exc)
    else:
        raise AssertionError("an error page must not parse")


def test_month_chunks_cover_the_range():
    assert iem._month_chunks(date(2025, 12, 15), date(2026, 1, 2)) == [
        (date(2025, 12, 1), date(2026, 1, 1)),
        (date(2026, 1, 1), date(2026, 2, 1)),
    ]
