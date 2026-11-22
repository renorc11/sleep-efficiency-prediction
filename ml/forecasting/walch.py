"""Walch et al. (PhysioNet) Apple Watch + PSG coming-night adapter.

Target: PSG-measured sleep efficiency of the single laboratory night,
``100 * sleep epochs / scored epochs`` (30-s epochs; label -1 = unscored and
excluded; 0 = wake; 1-5 = sleep stages).

Time is seconds relative to the PSG recording start (t = 0); there are no
calendar dates. The cutoff is 30 minutes before the PSG start, so no watch
data from the laboratory night is used. Each participant contributes one
target night, so only leave-one-participant-out validation is possible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import dataset_root, require_dir

DEFAULT_ROOT = (
    dataset_root()
    / "motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0"
)
TARGET = "psg_sleep_efficiency_pct"
CUTOFF_S = -30 * 60
DAY_S = 24 * 3600
EVENING_S = 3 * 3600


def psg_efficiency(labels: np.ndarray) -> tuple[float, int]:
    scored = labels[labels != -1]
    if len(scored) == 0:
        return float("nan"), 0
    return float(100.0 * (scored > 0).mean()), int(len(scored))


def window_stats(t: np.ndarray, v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (t >= lo) & (t < hi)
    return v[mask]


def subject_features(hr: pd.DataFrame, steps: pd.DataFrame, cutoff: float = CUTOFF_S) -> dict:
    """Pre-night features. ``hr``/``steps`` have columns t (s) and value."""
    assert (cutoff <= 0), "cutoff must precede the PSG start"
    t, v = hr["t"].to_numpy(), hr["value"].to_numpy(dtype=float)
    day = window_stats(t, v, cutoff - DAY_S, cutoff)
    eve = window_stats(t, v, cutoff - EVENING_S, cutoff)
    prior = window_stats(t, v, cutoff - 7 * DAY_S, cutoff - DAY_S)
    day_minutes = len(np.unique(np.floor(t[(t >= cutoff - DAY_S) & (t < cutoff)] / 60)))
    feats = {
        "hr_day_coverage": day_minutes / (DAY_S / 60),
        "hr_day_mean": day.mean() if len(day) >= 60 else np.nan,
        "hr_day_p05": np.percentile(day, 5) if len(day) >= 60 else np.nan,
        "hr_day_sd": day.std() if len(day) >= 60 else np.nan,
        "hr_evening_mean": eve.mean() if len(eve) >= 30 else np.nan,
        "hr_prior_days_mean": prior.mean() if len(prior) >= 60 else np.nan,
        "hr_prior_days_p05": np.percentile(prior, 5) if len(prior) >= 60 else np.nan,
        "hr_history_days": float(max(0.0, (cutoff - t.min()) / DAY_S)) if len(t) else 0.0,
    }
    st, sv = steps["t"].to_numpy(), steps["value"].to_numpy(dtype=float)
    s_day = window_stats(st, sv, cutoff - DAY_S, cutoff)
    s_eve = window_stats(st, sv, cutoff - EVENING_S, cutoff)
    feats.update(
        steps_day_bins=len(s_day),  # 10-minute bins observed out of 144
        steps_day=s_day.sum() if len(s_day) else np.nan,
        steps_evening=s_eve.sum() if len(s_eve) else np.nan,
    )
    return feats


def _read_pair(path: Path, sep=None) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=sep or r"[,\s]+", header=None, engine="python", names=["t", "value"])
    return frame.dropna()


def build_feature_table(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    require_dir(root / "labels", "Walch PSG labels folder")
    rows = []
    for label_file in sorted((root / "labels").glob("*_labeled_sleep.txt")):
        sid = label_file.name.split("_")[0]
        labels = _read_pair(label_file)
        se, scored = psg_efficiency(labels["value"].to_numpy(dtype=int))
        hr = _read_pair(root / "heart_rate" / f"{sid}_heartrate.txt")
        steps_path = root / "steps" / f"{sid}_steps.txt"
        steps = _read_pair(steps_path) if steps_path.exists() else pd.DataFrame({"t": [], "value": []})
        rows.append({"subject": sid, TARGET: se, "scored_epochs": scored, **subject_features(hr, steps)})
    return pd.DataFrame(rows)


FEATURES = [
    "hr_day_coverage", "hr_day_mean", "hr_day_p05", "hr_day_sd", "hr_evening_mean",
    "hr_prior_days_mean", "hr_prior_days_p05", "hr_history_days",
    "steps_day_bins", "steps_day", "steps_evening",
]
