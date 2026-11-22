"""StudentLife (Dartmouth 2013) coming-night adapter.

Target: next-night SELF-REPORTED sleep hours from the Sleep EMA ("How many
hours did you sleep last night?"). This is not sleep efficiency and is never
pooled with efficiency targets.

* Timestamps are Unix seconds, converted to America/New_York (the study site);
  the time zone is an assumption documented in the report.
* A report answered at local time T describes the night ending on T's date
  (or the previous date if T is before 04:00). Night date N = that wake date - 1.
* Duplicate reports for the same night: the latest answer is kept.
* Hour code 1 means "< 3 h" and is mapped to 3.0 with ``censored_low`` = 1;
  codes 2..19 map to 3.5..12.0 h. Codes outside 1..19 are dropped.
* Stress EMA is recoded to increase with stress:
  5 Feeling great -> 1, 4 Feeling good -> 2, 1 A little stressed -> 3,
  2 Definitely stressed -> 4, 3 Stressed out -> 5.
* Features for night N use only data before 21:00 local on N. Phone windows
  with < 50 % sensing coverage give NaN, never zero.
* Phone lock/dark/charge intervals are capped at 600 min by the collection
  software; the longest overnight interval is flagged when it hits the cap.
"""

from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .common import overlap_minutes
from .datasets import dataset_root, require_dir

DEFAULT_ROOT = dataset_root() / "dataset"
TZ = ZoneInfo("America/New_York")
TARGET = "self_reported_sleep_hours"
CUTOFF_HOUR = 21
DAY_START_HOUR = 6
CAP_MIN = 600
MIN_COVERAGE = 0.5
STRESS_RECODE = {5: 1, 4: 2, 1: 3, 2: 4, 3: 5}


def decode_hours(code) -> tuple[float, int] | None:
    try:
        code = int(code)
    except (TypeError, ValueError):
        return None
    if code == 1:
        return 3.0, 1
    if 2 <= code <= 19:
        return 3.5 + 0.5 * (code - 2), 0
    return None


def local(ts) -> pd.Timestamp:
    return pd.Timestamp(int(ts), unit="s", tz="UTC").tz_convert(TZ)


def parse_sleep_ema(records: list[dict], uid: str) -> pd.DataFrame:
    rows = []
    for r in records:
        if "hour" not in r or "resp_time" not in r:
            continue
        decoded = decode_hours(r.get("hour"))
        if decoded is None:
            continue
        t = local(r["resp_time"])
        wake = t.normalize() if t.hour >= 4 else t.normalize() - pd.Timedelta(days=1)
        try:
            rate = int(r.get("rate"))
            rate = rate if 1 <= rate <= 4 else np.nan
        except (TypeError, ValueError):
            rate = np.nan
        rows.append({"uid": uid, "resp_time": t, "night_date": wake - pd.Timedelta(days=1),
                     "hours": decoded[0], "censored_low": decoded[1], "rate": rate})
    frame = pd.DataFrame(rows, columns=["uid", "resp_time", "night_date", "hours", "censored_low", "rate"])
    return frame.sort_values("resp_time").groupby("night_date").tail(1).sort_values("night_date")


def parse_stress_ema(records: list[dict], uid: str) -> pd.DataFrame:
    rows = []
    for r in records:
        try:
            level = STRESS_RECODE[int(r["level"])]
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({"uid": uid, "resp_time": local(r["resp_time"]), "stress": level})
    return pd.DataFrame(rows, columns=["uid", "resp_time", "stress"])


def read_intervals(path: Path, merge: bool = True) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 2))
    frame = pd.read_csv(path)
    frame.columns = [c.strip() for c in frame.columns]
    raw = frame[["start", "end"]].to_numpy(dtype=float)
    return merge_intervals(raw) if merge else raw


def merge_intervals(iv: np.ndarray) -> np.ndarray:
    """Union of possibly overlapping [start, end) intervals.

    Source files contain overlapping intervals; summing them unmerged counted
    the same minutes twice (window fractions above 1).
    """
    iv = iv[np.isfinite(iv).all(axis=1) & (iv[:, 1] > iv[:, 0])]
    if len(iv) == 0:
        return np.empty((0, 2))
    iv = iv[np.argsort(iv[:, 0])]
    merged = [iv[0].copy()]
    for s, e in iv[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append(np.array([s, e]))
    return np.vstack(merged)


def sensing_hours(path: Path) -> np.ndarray:
    """Unix hours with at least one activity-inference sample (coverage proxy)."""
    if not path.exists():
        return np.empty(0)
    ts = pd.read_csv(path, usecols=[0]).iloc[:, 0].to_numpy(dtype=float)
    return np.unique(np.floor(ts / 3600))


def coverage(hours: np.ndarray, lo: float, hi: float) -> float:
    lo_h, hi_h = np.floor(lo / 3600), np.ceil(hi / 3600)
    expected = hi_h - lo_h
    return float(((hours >= lo_h) & (hours < hi_h)).sum() / expected) if expected > 0 else 0.0


def phone_features(night: pd.Timestamp, lock, dark, charge, hours, lock_raw=None) -> dict:
    lock_raw = lock if lock_raw is None else lock_raw
    cutoff = (night + pd.Timedelta(hours=CUTOFF_HOUR)).timestamp()
    day_lo = (night + pd.Timedelta(hours=DAY_START_HOUR)).timestamp()
    eve_lo = cutoff - 3 * 3600
    prev_lo = (night - pd.Timedelta(days=1) + pd.Timedelta(hours=CUTOFF_HOUR)).timestamp()
    prev_hi = (night + pd.Timedelta(hours=12)).timestamp()
    feats = {}
    for label, lo, hi in (("day", day_lo, cutoff), ("evening", eve_lo, cutoff), ("prev_night", prev_lo, prev_hi)):
        cov = coverage(hours, lo, hi)
        feats[f"sensing_{label}_coverage"] = cov
        ok = cov >= MIN_COVERAGE
        window = (hi - lo) / 60
        for name, iv in (("locked", lock), ("dark", dark), ("charging", charge)):
            minutes = overlap_minutes(iv[:, 0], iv[:, 1], lo, hi) if len(iv) else 0.0
            feats[f"{name}_{label}_frac"] = minutes / window if ok else np.nan
        starts = ((lock[:, 0] >= lo) & (lock[:, 0] < hi)).sum() if len(lock) else 0
        feats[f"lock_events_{label}"] = float(starts) if ok else np.nan
    if feats["sensing_prev_night_coverage"] >= MIN_COVERAGE and len(lock):
        inside = lock[(lock[:, 1] > prev_lo) & (lock[:, 0] < prev_hi)]
        longest = float(((np.minimum(inside[:, 1], prev_hi) - np.maximum(inside[:, 0], prev_lo)) / 60).max()) if len(inside) else 0.0
        raw = lock_raw[(lock_raw[:, 1] > prev_lo) & (lock_raw[:, 0] < prev_hi)]
        capped = bool(len(raw) and ((raw[:, 1] - raw[:, 0]) / 60 >= CAP_MIN - 1).any())
        feats["prev_night_longest_lock_min"] = longest
        feats["prev_night_lock_capped"] = float(capped)
    else:
        feats["prev_night_longest_lock_min"] = np.nan
        feats["prev_night_lock_capped"] = np.nan
    return feats


def history_features(target_night: pd.Timestamp, cutoff: pd.Timestamp, sleep: pd.DataFrame) -> dict:
    """Earlier reports of the same student that were submitted before the cutoff."""
    known = sleep[(sleep["night_date"] < target_night) & (sleep["resp_time"] < cutoff)]
    prev = known[known["night_date"] == target_night - pd.Timedelta(days=1)]
    last7 = known[known["night_date"] >= target_night - pd.Timedelta(days=7)]
    return {
        "prev_available": int(len(prev) > 0),
        "prev_sleep_hours": float(prev["hours"].iloc[-1]) if len(prev) else np.nan,
        "prev_sleep_rate": float(prev["rate"].iloc[-1]) if len(prev) else np.nan,
        "hist_mean": float(known["hours"].mean()) if len(known) else np.nan,
        "hist_mean_last7": float(last7["hours"].mean()) if len(last7) else np.nan,
        "hist_count": float(len(known)),
    }


def stress_features(night: pd.Timestamp, cutoff: pd.Timestamp, stress: pd.DataFrame) -> dict:
    day = stress[(stress["resp_time"] >= night) & (stress["resp_time"] < cutoff)]
    return {"stress_day_mean": float(day["stress"].mean()) if len(day) else np.nan, "stress_day_count": float(len(day))}


FEATURES = [
    "prev_available", "prev_sleep_hours", "prev_sleep_rate", "hist_mean", "hist_mean_last7", "hist_count",
    "stress_day_mean", "stress_day_count", "weekend_night",
] + [f"{n}_{w}_frac" for w in ("day", "evening", "prev_night") for n in ("locked", "dark", "charging")] + [
    "lock_events_day", "lock_events_evening", "lock_events_prev_night",
    "sensing_day_coverage", "sensing_evening_coverage", "sensing_prev_night_coverage",
    "prev_night_longest_lock_min", "prev_night_lock_capped",
]


def build_feature_table(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    require_dir(root / "EMA" / "response" / "Sleep", "StudentLife Sleep EMA folder")
    rows = []
    for sleep_file in sorted((root / "EMA" / "response" / "Sleep").glob("Sleep_u*.json")):
        uid = sleep_file.stem.split("_")[1]
        sleep = parse_sleep_ema(json.loads(sleep_file.read_text()), uid)
        stress_file = root / "EMA" / "response" / "Stress" / f"Stress_{uid}.json"
        stress = parse_stress_ema(json.loads(stress_file.read_text()) if stress_file.exists() else [], uid)
        sensing = root / "sensing"
        lock = read_intervals(sensing / "phonelock" / f"phonelock_{uid}.csv")
        dark = read_intervals(sensing / "dark" / f"dark_{uid}.csv")
        charge = read_intervals(sensing / "phonecharge" / f"phonecharge_{uid}.csv")
        lock_raw = read_intervals(sensing / "phonelock" / f"phonelock_{uid}.csv", merge=False)
        hours = sensing_hours(sensing / "activity" / f"activity_{uid}.csv")
        for r in sleep.itertuples():
            night = r.night_date
            cutoff = night + pd.Timedelta(hours=CUTOFF_HOUR)
            rows.append({
                "uid": uid, "night_date": night, TARGET: r.hours, "target_censored_low": r.censored_low,
                **history_features(night, cutoff, sleep),
                **stress_features(night, cutoff, stress),
                **phone_features(night, lock, dark, charge, hours, lock_raw),
                "weekend_night": int(night.dayofweek in (4, 5)),
            })
    table = pd.DataFrame(rows)
    table["night_order"] = table["night_date"].dt.tz_localize(None).astype("int64") // 86_400_000_000_000
    return table
