"""Fitabase (FitBit Fitness Tracker, MTurk 2016) coming-night adapter.

Target: next-night device-derived sleep ratio, ``100 * asleep minutes / session
minutes`` of the night's main sleep. This is a consumer-device proxy, not
PSG-measured sleep efficiency and not Fitbit's own efficiency field.

Night definition (applied identically to every row):
* A sleep session is one Fitbit ``logId``. Duplicated minute rows are removed.
* Night date D = calendar date of (onset - 12 h), so a 02:00 onset on D+1
  belongs to night D.
* Main sleep = the longest session of night D with 180 <= minutes <= 900 and
  onset at or after the cutoff on D. Earlier-onset sleeps cannot be forecast
  at that cutoff and are excluded as targets (they remain usable as history).

Features for night D use only timestamps strictly before ``D + cutoff``.
Timestamps are local wall-clock time without a time zone, as exported.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .common import add_history_features
from .datasets import dataset_root, require_dir

DEFAULT_ROOT = dataset_root() / "archive (8)"
TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"
TARGET = "sleep_ratio_proxy_pct"
MIN_MAIN_SLEEP, MAX_MAIN_SLEEP = 180, 900
EVENING_HOURS = 3


def _read_all(root: Path, name: str, usecols=None) -> pd.DataFrame:
    require_dir(root, "Fitabase folder ('archive (8)')")
    paths = sorted(glob.glob(str(root / "*" / "*" / name)))
    if not paths:
        raise FileNotFoundError(f"{name} not found under {root}")
    return pd.concat([pd.read_csv(p, usecols=usecols) for p in paths], ignore_index=True)


def load_sessions(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    raw = _read_all(root, "minuteSleep_merged.csv")
    return sessions_from_minutes(raw)


def sessions_from_minutes(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.drop_duplicates().copy()
    raw["t"] = pd.to_datetime(raw["date"], format=TIME_FORMAT) if raw["date"].dtype == object else raw["date"]
    raw = raw.drop_duplicates(subset=["Id", "logId", "t"])
    agg = raw.groupby(["Id", "logId"]).agg(
        onset=("t", "min"), last=("t", "max"), minutes=("t", "size"),
        asleep_min=("value", lambda v: int((v == 1).sum())),
        restless_min=("value", lambda v: int((v == 2).sum())),
        awake_min=("value", lambda v: int((v == 3).sum())),
    ).reset_index()
    agg["offset"] = agg["last"] + pd.Timedelta(minutes=1)
    agg["span_min"] = (agg["offset"] - agg["onset"]).dt.total_seconds() / 60
    # Minute rows are contiguous in this export; a gap would mean the span
    # overstates time in bed, so the ratio uses observed minutes only.
    agg["ratio_pct"] = 100.0 * agg["asleep_min"] / agg["minutes"]
    agg["night_date"] = (agg["onset"] - pd.Timedelta(hours=12)).dt.normalize()
    return agg.drop(columns="last")


def main_sleeps(sessions: pd.DataFrame, cutoff_hour: float) -> pd.DataFrame:
    s = sessions[(sessions["minutes"] >= MIN_MAIN_SLEEP) & (sessions["minutes"] <= MAX_MAIN_SLEEP)].copy()
    per_night = s.groupby(["Id", "night_date"])["logId"].transform("size")
    s["sessions_in_night"] = per_night
    s = s.sort_values("minutes").groupby(["Id", "night_date"]).tail(1)
    s["cutoff"] = s["night_date"] + pd.to_timedelta(cutoff_hour, unit="h")
    s["forecastable"] = s["onset"] >= s["cutoff"]
    return s.sort_values(["Id", "night_date"]).reset_index(drop=True)


class _Series:
    """Per-participant cumulative sums for O(log n) window queries."""

    def __init__(self, times: np.ndarray, values: np.ndarray):
        order = np.argsort(times)
        self.t = times[order]
        self.cum = np.concatenate([[0.0], np.cumsum(values[order])])

    def window(self, lo: np.datetime64, hi: np.datetime64) -> tuple[float, int]:
        a, b = np.searchsorted(self.t, lo, "left"), np.searchsorted(self.t, hi, "left")
        return float(self.cum[b] - self.cum[a]), int(b - a)


def _series_by_id(frame: pd.DataFrame, time_col: str, value: np.ndarray) -> dict:
    out = {}
    for pid, idx in frame.groupby("Id").indices.items():
        out[pid] = _Series(frame[time_col].to_numpy()[idx], value[idx])
    return out


def load_minute_activity(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    steps = _read_all(root, "minuteStepsNarrow_merged.csv")
    mets = _read_all(root, "minuteMETsNarrow_merged.csv")
    for f, col in ((steps, "ActivityMinute"), (mets, "ActivityMinute")):
        f["t"] = pd.to_datetime(f[col], format=TIME_FORMAT)
    steps = steps.drop_duplicates(subset=["Id", "t"])
    mets = mets.drop_duplicates(subset=["Id", "t"])
    return steps[["Id", "t", "Steps"]].merge(mets[["Id", "t", "METs"]], on=["Id", "t"], how="outer")


def load_minute_heart_rate(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    hr = _read_all(root, "heartrate_seconds_merged.csv")
    hr["t"] = pd.to_datetime(hr["Time"], format=TIME_FORMAT).dt.floor("min")
    hr = hr.drop_duplicates(subset=["Id", "Time"])
    return hr.groupby(["Id", "t"], as_index=False)["Value"].mean()


def activity_features(nights: pd.DataFrame, activity: pd.DataFrame) -> pd.DataFrame:
    """Day (00:00 to cutoff) and evening (last 3 h before cutoff) activity.

    Windows with no minute rows give NaN values and zero coverage. Zero steps
    in a recorded minute is kept as zero, but Fitabase cannot separate
    "not moving" from "not worn"; that limitation is reported, not imputed.
    """
    activity = activity.copy()
    activity["step_min"] = (activity["Steps"].fillna(0) > 0).astype(float)
    activity["mvpa_min"] = (activity["METs"].fillna(0) >= 30).astype(float)  # METs exported x10
    activity["row"] = 1.0
    by_value = {
        c: _series_by_id(activity, "t", activity[c].fillna(0).to_numpy(dtype=float))
        for c in ("Steps", "step_min", "mvpa_min", "row")
    }
    rows = []
    for r in nights.itertuples():
        day_lo, hi = r.night_date.to_datetime64(), r.cutoff.to_datetime64()
        eve_lo = (r.cutoff - pd.Timedelta(hours=EVENING_HOURS)).to_datetime64()
        feats = {}
        series = {c: by_value[c].get(r.Id) for c in by_value}
        for label, lo in (("day", day_lo), ("evening", eve_lo)):
            expected = (hi - lo) / np.timedelta64(1, "m")
            if series["row"] is None:
                observed = 0
            else:
                _, observed = series["row"].window(lo, hi)
            feats[f"activity_{label}_coverage"] = observed / expected
            for c, name in (("Steps", "steps"), ("step_min", "stepping_minutes"), ("mvpa_min", "mvpa_minutes")):
                feats[f"{name}_{label}"] = series[c].window(lo, hi)[0] if observed else np.nan
        rows.append(feats)
    return pd.DataFrame(rows, index=nights.index)


def heart_rate_features(nights: pd.DataFrame, hr_minutes: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    by_id = {pid: g.sort_values("t") for pid, g in hr_minutes.groupby("Id")}
    rows = []
    for r in nights.itertuples():
        feats = {
            "hr_day_mean": np.nan, "hr_day_p05": np.nan, "hr_evening_mean": np.nan,
            "hr_prev_sleep_mean": np.nan, "hr_day_coverage": 0.0,
        }
        g = by_id.get(r.Id)
        if g is not None:
            day = g[(g["t"] >= r.night_date) & (g["t"] < r.cutoff)]["Value"]
            eve = g[(g["t"] >= r.cutoff - pd.Timedelta(hours=EVENING_HOURS)) & (g["t"] < r.cutoff)]["Value"]
            expected = (r.cutoff - r.night_date).total_seconds() / 60
            feats["hr_day_coverage"] = len(day) / expected
            if len(day) >= 60:  # at least an hour of minutes before summarising
                feats["hr_day_mean"] = day.mean()
                feats["hr_day_p05"] = day.quantile(0.05)
            if len(eve) >= 30:
                feats["hr_evening_mean"] = eve.mean()
            prev = previous.loc[r.Index]
            if prev["prev_available"]:
                sl = g[(g["t"] >= prev["prev_onset"]) & (g["t"] < prev["prev_offset"])]["Value"]
                if len(sl) >= 60:
                    feats["hr_prev_sleep_mean"] = sl.mean()
        rows.append(feats)
    return pd.DataFrame(rows, index=nights.index)


def previous_night_features(nights: pd.DataFrame, all_main: pd.DataFrame) -> pd.DataFrame:
    """Main sleep of night D-1 (any onset time), which ends before D's cutoff."""
    key = all_main.set_index(["Id", "night_date"])
    rows = []
    for r in nights.itertuples():
        prev_date = r.night_date - pd.Timedelta(days=1)
        feats = {"prev_available": False}
        if (r.Id, prev_date) in key.index:
            p = key.loc[(r.Id, prev_date)]
            if p["offset"] <= r.cutoff:
                feats.update(
                    prev_available=True, prev_onset=p["onset"], prev_offset=p["offset"],
                    prev_minutes=p["minutes"], prev_asleep_min=p["asleep_min"], prev_ratio_pct=p["ratio_pct"],
                    prev_awake_min=p["awake_min"], prev_restless_min=p["restless_min"],
                    prev_onset_hour=(p["onset"] - prev_date).total_seconds() / 3600,
                    prev_wake_hour=(p["offset"] - r.night_date).total_seconds() / 3600,
                )
        rows.append(feats)
    out = pd.DataFrame(rows, index=nights.index)
    for c in ("prev_minutes", "prev_asleep_min", "prev_ratio_pct", "prev_awake_min", "prev_restless_min",
              "prev_onset_hour", "prev_wake_hour"):
        if c not in out:
            out[c] = np.nan
    return out


def nap_features(nights: pd.DataFrame, sessions: pd.DataFrame) -> pd.Series:
    """Minutes of short sleeps (< 180 min) that start and end on day D before the cutoff."""
    naps = sessions[sessions["minutes"] < MIN_MAIN_SLEEP]
    out = []
    for r in nights.itertuples():
        n = naps[(naps["Id"] == r.Id) & (naps["onset"] >= r.night_date + pd.Timedelta(hours=8)) & (naps["offset"] <= r.cutoff)]
        out.append(float(n["minutes"].sum()))
    return pd.Series(out, index=nights.index, name="nap_minutes_day")


FEATURES = [
    "prev_available", "prev_minutes", "prev_asleep_min", "prev_ratio_pct", "prev_awake_min", "prev_restless_min",
    "prev_onset_hour", "prev_wake_hour",
    "hist_mean", "hist_mean_last7", "hist_count", "nights_observed_last7",
    "steps_day", "stepping_minutes_day", "mvpa_minutes_day", "activity_day_coverage",
    "steps_evening", "stepping_minutes_evening", "mvpa_minutes_evening", "activity_evening_coverage",
    "hr_day_mean", "hr_day_p05", "hr_evening_mean", "hr_prev_sleep_mean", "hr_day_coverage",
    "nap_minutes_day", "weekend_night",
]


def build_feature_table(cutoff_hour: float = 20.0, root: Path = DEFAULT_ROOT,
                        sessions=None, activity=None, hr_minutes=None) -> pd.DataFrame:
    sessions = load_sessions(root) if sessions is None else sessions
    activity = load_minute_activity(root) if activity is None else activity
    hr_minutes = load_minute_heart_rate(root) if hr_minutes is None else hr_minutes

    all_main = main_sleeps(sessions, cutoff_hour)
    # History uses every earlier main sleep, forecastable or not; each one ended
    # before the current night's cutoff.
    all_main = add_history_features(all_main, "Id", "night_date", "ratio_pct")
    dates = all_main.groupby("Id")["night_date"]
    all_main["nights_observed_last7"] = [
        int(((d - pd.Timedelta(days=7) <= g) & (g < d)).sum())
        for d, g in zip(all_main["night_date"], (dates.get_group(i) for i in all_main["Id"]))
    ]
    nights = all_main[all_main["forecastable"]].copy()
    prev = previous_night_features(nights, all_main)
    nights = pd.concat([nights, prev.drop(columns=["prev_onset", "prev_offset"], errors="ignore")], axis=1)
    nights = pd.concat([nights, activity_features(nights, activity)], axis=1)
    nights = pd.concat([nights, heart_rate_features(nights, hr_minutes, prev)], axis=1)
    nights["nap_minutes_day"] = nap_features(nights, sessions)
    nights["weekend_night"] = nights["night_date"].dt.dayofweek.isin([4, 5]).astype(int)
    nights["prev_available"] = nights["prev_available"].astype(int)
    nights[TARGET] = nights["ratio_pct"]
    nights["night_order"] = nights["night_date"].astype("int64") // 86_400_000_000_000
    keep = ["Id", "night_date", "night_order", "cutoff", "onset", "sessions_in_night", TARGET] + FEATURES
    return nights[keep].reset_index(drop=True)
