"""Tests for the coming-night forecasting package (ml/forecasting).

Unit tests use small hand-built frames. The dataset smoke tests run only when
the local research datasets are present.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.forecasting import common, fitabase, studentlife, walch  # noqa: E402

T = pd.Timestamp


# ---------- Fitabase ----------

def _minutes(pid, log_id, start, values):
    times = pd.date_range(start, periods=len(values), freq="min")
    return pd.DataFrame({"Id": pid, "date": times, "value": values, "logId": log_id})


def _sessions():
    raw = pd.concat([
        _minutes(1, 10, "2016-04-01 23:00", [1] * 400 + [3] * 20),   # night 04-01
        _minutes(1, 11, "2016-04-03 01:30", [1] * 300 + [2] * 60),   # onset after midnight -> night 04-02
        _minutes(1, 12, "2016-04-03 20:30", [1] * 360),              # night 04-03, onset 20:30
        _minutes(1, 13, "2016-04-03 14:00", [1] * 40),               # nap on 04-03
    ])
    return fitabase.sessions_from_minutes(pd.concat([raw, raw.iloc[:5]]))  # duplicated rows


def test_sessions_deduplicate_and_assign_night_date():
    s = _sessions().set_index("logId")
    assert s.loc[10, "minutes"] == 420  # duplicates removed
    assert s.loc[10, "ratio_pct"] == pytest.approx(100 * 400 / 420)
    assert s.loc[11, "night_date"] == T("2016-04-02")
    assert s.loc[10, "night_date"] == T("2016-04-01")


def test_cutoff_controls_which_nights_are_forecastable():
    s = _sessions()
    at20 = fitabase.main_sleeps(s, 20.0).set_index("night_date")
    at21 = fitabase.main_sleeps(s, 21.0).set_index("night_date")
    assert at20.loc[T("2016-04-03"), "forecastable"]
    assert not at21.loc[T("2016-04-03"), "forecastable"]  # 20:30 onset precedes a 21:00 cutoff
    assert 13 not in set(at20["logId"])  # naps are never main sleeps


def test_previous_night_uses_only_night_before():
    main = fitabase.main_sleeps(_sessions(), 20.0)
    target = main[main["night_date"] == T("2016-04-03")]
    prev = fitabase.previous_night_features(target, main).iloc[0]
    assert prev["prev_available"]
    assert prev["prev_minutes"] == 360
    assert prev["prev_offset"] <= target["cutoff"].iloc[0]
    first = main[main["night_date"] == T("2016-04-01")]
    assert not fitabase.previous_night_features(first, main).iloc[0]["prev_available"]


def test_activity_ignores_data_after_cutoff_and_keeps_missing_as_nan():
    night = pd.DataFrame({"Id": [1, 2], "night_date": [T("2016-04-03")] * 2, "cutoff": [T("2016-04-03 20:00")] * 2})
    times = pd.date_range("2016-04-03 00:00", "2016-04-03 23:59", freq="min")
    steps = np.where(times < T("2016-04-03 20:00"), 1, 1000)  # huge values after the cutoff
    activity = pd.DataFrame({"Id": 1, "t": times, "Steps": steps, "METs": 10})
    feats = fitabase.activity_features(night, activity)
    assert feats.loc[0, "steps_day"] == 20 * 60
    assert feats.loc[0, "steps_evening"] == 3 * 60
    assert feats.loc[0, "activity_day_coverage"] == pytest.approx(1.0)
    assert np.isnan(feats.loc[1, "steps_day"])  # participant 2 has no data: NaN, not 0
    assert feats.loc[1, "activity_day_coverage"] == 0


def test_heart_rate_missing_participant_gives_nan():
    night = pd.DataFrame({"Id": [5], "night_date": [T("2016-04-03")], "cutoff": [T("2016-04-03 20:00")]})
    prev = pd.DataFrame({"prev_available": [False]})
    feats = fitabase.heart_rate_features(night, pd.DataFrame(columns=["Id", "t", "Value"]), prev).iloc[0]
    assert np.isnan(feats["hr_day_mean"]) and feats["hr_day_coverage"] == 0


def test_history_features_exclude_current_night():
    frame = pd.DataFrame({"pid": [1, 1, 1, 2], "d": [1, 2, 3, 1], "y": [80.0, 90.0, 1000.0, 70.0]})
    out = common.add_history_features(frame, "pid", "d", "y").set_index(["pid", "d"])
    assert np.isnan(out.loc[(1, 1), "hist_mean"])
    assert out.loc[(1, 3), "hist_mean"] == 85.0  # the 1000 of night 3 is not used
    assert np.isnan(out.loc[(2, 1), "hist_mean"])  # no cross-participant history


# ---------- Walch ----------

def test_psg_efficiency_excludes_unscored_epochs():
    se, scored = walch.psg_efficiency(np.array([-1, -1, 0, 0, 2, 2, 2, 5]))
    assert scored == 6 and se == pytest.approx(100 * 4 / 6)


def test_walch_features_ignore_laboratory_night():
    t = np.arange(-2 * 86400, 8 * 3600, 60.0)
    hr = pd.DataFrame({"t": t, "value": np.where(t < walch.CUTOFF_S, 60.0, 150.0)})
    steps = pd.DataFrame({"t": np.arange(-86400, 0, 600.0), "value": 10.0})
    feats = walch.subject_features(hr, steps)
    assert feats["hr_day_mean"] == 60.0
    with pytest.raises(AssertionError):
        walch.subject_features(hr, steps, cutoff=60)


# ---------- StudentLife ----------

def test_hour_codes_and_stress_recode():
    assert studentlife.decode_hours("1") == (3.0, 1)
    assert studentlife.decode_hours(2) == (3.5, 0)
    assert studentlife.decode_hours(19) == (12.0, 0)
    assert studentlife.decode_hours(25) is None and studentlife.decode_hours("x") is None
    stress = studentlife.parse_stress_ema([{"level": "3", "resp_time": 1365600000}, {"level": "5", "resp_time": 1365600000}], "u")
    assert list(stress["stress"]) == [5, 1]  # stressed out -> 5, feeling great -> 1


def _unix(local_time):
    return int(pd.Timestamp(local_time, tz=studentlife.TZ).timestamp())


def test_sleep_report_night_assignment_timezone_and_duplicates():
    records = [
        {"hour": "11", "rate": "1", "resp_time": _unix("2013-04-10 08:00")},   # night 04-09
        {"hour": "9", "rate": "2", "resp_time": _unix("2013-04-10 10:00")},    # later duplicate wins
        {"hour": "7", "rate": "3", "resp_time": _unix("2013-04-12 02:00")},    # before 04:00 -> night 04-10
        {"hour": "40", "resp_time": _unix("2013-04-13 08:00")},                 # invalid code dropped
    ]
    s = studentlife.parse_sleep_ema(records, "u00")
    assert list(s["night_date"].dt.strftime("%m-%d")) == ["04-09", "04-10"]
    assert list(s["hours"]) == [7.0, 6.0]


def test_history_excludes_reports_submitted_after_cutoff():
    night = pd.Timestamp("2013-04-10", tz=studentlife.TZ)
    cutoff = night + pd.Timedelta(hours=21)
    sleep = pd.DataFrame({
        "night_date": [night - pd.Timedelta(days=2), night - pd.Timedelta(days=1), night],
        "resp_time": [night - pd.Timedelta(hours=12), cutoff + pd.Timedelta(hours=1), night + pd.Timedelta(days=1, hours=9)],
        "hours": [6.0, 9.0, 4.0], "rate": [2, 1, 3],
    })
    h = studentlife.history_features(night, cutoff, sleep)
    assert h["prev_available"] == 0  # last night's report arrived after the cutoff
    assert h["hist_mean"] == 6.0 and h["hist_count"] == 1


def test_merge_intervals_and_low_coverage_gives_nan():
    merged = studentlife.merge_intervals(np.array([[0, 10], [5, 20], [30, 40], [np.nan, 3]]))
    assert merged.tolist() == [[0, 20], [30, 40]]
    night = pd.Timestamp("2013-04-10", tz=studentlife.TZ)
    lock = np.array([[(night + pd.Timedelta(hours=7)).timestamp(), (night + pd.Timedelta(hours=8)).timestamp()]])
    no_sensing = studentlife.phone_features(night, lock, lock, lock, np.empty(0))
    assert np.isnan(no_sensing["locked_day_frac"]) and no_sensing["sensing_day_coverage"] == 0
    hours = np.arange(np.floor((night - pd.Timedelta(days=1)).timestamp() / 3600), np.ceil((night + pd.Timedelta(days=1)).timestamp() / 3600))
    full = studentlife.phone_features(night, lock, lock, lock, hours)
    assert full["locked_day_frac"] == pytest.approx(60 / (15 * 60))


def test_phone_features_ignore_intervals_after_cutoff():
    night = pd.Timestamp("2013-04-10", tz=studentlife.TZ)
    hours = np.arange(np.floor((night - pd.Timedelta(days=1)).timestamp() / 3600), np.ceil((night + pd.Timedelta(days=2)).timestamp() / 3600))
    late = np.array([[(night + pd.Timedelta(hours=21, minutes=30)).timestamp(), (night + pd.Timedelta(hours=23)).timestamp()]])
    feats = studentlife.phone_features(night, late, late, late, hours)
    assert feats["locked_evening_frac"] == 0 and feats["lock_events_day"] == 0


# ---------- evaluation ----------

def test_evaluation_rejects_participant_overlap():
    frame = pd.DataFrame({"pid": [1, 1, 2, 2], "x": [1.0, 2, 3, 4], "y": [1.0, 2, 3, 4]})
    with pytest.raises(AssertionError, match="participant leakage"):
        common.evaluate_grouped(frame, ["x"], "y", "pid", [(np.array([0, 2]), np.array([1, 3]))], None, None)


def test_grouped_evaluation_keeps_participants_separate_and_scores_baselines():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"pid": np.repeat(np.arange(10), 8), "x": rng.normal(size=80)})
    frame["y"] = 80 + 2 * frame["x"] + rng.normal(scale=0.1, size=80)
    frame["hist"] = 80.0
    from sklearn.model_selection import GroupKFold
    splits = list(GroupKFold(5).split(frame, groups=frame["pid"]))
    oof = common.evaluate_grouped(frame, ["x"], "y", "pid", splits, None, "hist")
    result = common.summarise(oof, "y", "pid", "t", "d", "v")
    assert result.n_participants == 10 and result.n_nights == 80
    assert result.metrics["random_forest"]["mae"] < result.metrics["baseline_training_mean"]["mae"]
    assert result.paired_vs_best_baseline["random_forest"]["mae_difference"] < 0


# ---------- smoke tests on the local datasets ----------

@pytest.mark.skipif(not fitabase.DEFAULT_ROOT.exists(), reason="Fitabase dataset not available")
def test_fitabase_table_respects_cutoff():
    table = fitabase.build_feature_table(20.0)
    assert (table["onset"] >= table["cutoff"]).all()
    assert (table["prev_wake_hour"].dropna() <= 20.0).all()
    assert table[fitabase.TARGET].between(0, 100).all()
    assert not table.duplicated(["Id", "night_date"]).any()


@pytest.mark.skipif(not studentlife.DEFAULT_ROOT.exists(), reason="StudentLife dataset not available")
def test_studentlife_table_fractions_are_bounded():
    table = studentlife.build_feature_table()
    fracs = table[[c for c in table if c.endswith("_frac")]]
    assert (fracs.fillna(0) <= 1.0 + 1e-9).all().all()
    assert not table.duplicated(["uid", "night_date"]).any()
