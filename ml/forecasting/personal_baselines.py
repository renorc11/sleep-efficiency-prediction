"""Personal-history forecasts of the coming night (forecasting_v2).

Every forecast uses only that participant's own earlier nights whose outcome
was available before the target night's cutoff. Nights without an
observation are skipped, never filled with zero; "previous k valid nights"
means the last k observed nights, whatever calendar span they cover.

Methods
* expanding_mean        - all earlier valid nights (the forecasting_v1 baseline)
* mean_last7, mean_last14
* ewma_h{3,7,14}        - exponentially weighted mean; half-life in valid nights
* trend_adjusted        - mean of the last 14 nights plus a damped linear trend,
                          only with >= 14 nights; otherwise the expanding mean.
                          The trend adjustment is limited to +/- 1 SD of the
                          window and the result clipped to the target's range.

The decision rule is fixed here, before results are inspected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_HISTORY = 3
EWMA_HALF_LIVES = (3, 7, 14)
TREND_MIN_HISTORY = 14
TREND_WINDOW = 14
TREND_DAMPING = 0.5
INTERVAL_Z = 1.2816  # central 80 % under a normal approximation

# A method replaces the expanding mean only if, for that dataset, the paired
# participant-bootstrap 95 % CI of MAE(method) - MAE(expanding_mean) lies
# entirely below zero in BOTH the participant-independent and the
# chronological evaluation, AND the point improvement is >= 2 % of the
# baseline MAE. Otherwise the simpler expanding mean is retained.
MIN_RELATIVE_IMPROVEMENT = 0.02
# Correction made after the first run, before any method was adopted: fixed
# half-lives (ewma_h3/h7/h14) are reported descriptively only. Letting them
# compete would pick a half-life on the evaluated participants/dates; the
# only EWMA candidate is ewma_selected, whose half-life is chosen on the
# training part of each split.
DECISION_CANDIDATES = ("mean_last7", "mean_last14", "trend_adjusted", "ewma_selected")

FIXED_METHODS = ["expanding_mean", "mean_last7", "mean_last14"] + [f"ewma_h{h}" for h in EWMA_HALF_LIVES] + ["trend_adjusted"]


def ewma(values: np.ndarray, half_life: float) -> float:
    """Weights 0.5 ** (age / half_life); age 0 is the most recent night."""
    age = np.arange(len(values))[::-1]
    w = 0.5 ** (age / half_life)
    return float(np.sum(w * values) / np.sum(w))


def trend_adjusted(values: np.ndarray, days: np.ndarray, target_day: float, bounds: tuple[float, float]) -> float:
    if len(values) < TREND_MIN_HISTORY:
        return float(values.mean())
    v, d = values[-TREND_WINDOW:], days[-TREND_WINDOW:].astype(float)
    mean = v.mean()
    if np.ptp(d) == 0:
        return float(mean)
    slope = np.polyfit(d, v, 1)[0]
    adjustment = TREND_DAMPING * slope * (target_day - d.mean())
    sd = v.std(ddof=1)
    adjustment = float(np.clip(adjustment, -sd, sd))
    return float(np.clip(mean + adjustment, *bounds))


def forecast_from_history(values: np.ndarray, days: np.ndarray, target_day: float, bounds: tuple[float, float]) -> dict:
    """All methods for one target night. ``values``/``days`` are earlier valid nights in time order."""
    n = len(values)
    out = {"history_count": n}
    if n < MIN_HISTORY:
        return out
    out.update({
        "expanding_mean": float(values.mean()),
        "mean_last7": float(values[-7:].mean()),
        "mean_last14": float(values[-14:].mean()),
        "trend_adjusted": trend_adjusted(values, days, target_day, bounds),
        "trend_applied": int(n >= TREND_MIN_HISTORY),
        "history_sd": float(values.std(ddof=1)),
        "history_sd_last7": float(values[-7:].std(ddof=1)) if n >= 2 else np.nan,
        "history_span_days": float(target_day - days[0]),
        "days_since_last": float(target_day - days[-1]),
    })
    for h in EWMA_HALF_LIVES:
        out[f"ewma_h{h}"] = ewma(values, h)
    return out


@dataclass
class PersonalTask:
    dataset: str
    target: str
    targets: pd.DataFrame      # pid, night_day, cutoff, y
    history: pd.DataFrame      # pid, night_day, available_at, value
    bounds: tuple[float, float]


def build_forecasts(task: PersonalTask) -> pd.DataFrame:
    """One row per target night, with history taken strictly before its cutoff."""
    by_pid = {pid: g.sort_values("night_day") for pid, g in task.history.groupby("pid")}
    rows = []
    for r in task.targets.itertuples():
        h = by_pid.get(r.pid)
        if h is None:
            h = task.history.iloc[0:0]
        h = h[(h["night_day"] < r.night_day) & (h["available_at"] < r.cutoff)]
        # One observation per night: duplicates are resolved upstream; assert it.
        assert not h["night_day"].duplicated().any(), "duplicate history nights"
        rows.append({"pid": r.pid, "night_day": r.night_day, "y": r.y,
                     **forecast_from_history(h["value"].to_numpy(float), h["night_day"].to_numpy(float),
                                             float(r.night_day), task.bounds)})
    return pd.DataFrame(rows)


# ---------- dataset adapters ----------

def fitabase_task(cutoff_hour: float = 20.0) -> PersonalTask:
    from . import fitabase

    main = fitabase.main_sleeps(fitabase.load_sessions(), cutoff_hour)
    day = (main["night_date"] - pd.Timestamp("1970-01-01")).dt.days
    history = pd.DataFrame({"pid": main["Id"], "night_day": day, "available_at": main["offset"], "value": main["ratio_pct"]})
    t = main[main["forecastable"]]
    targets = pd.DataFrame({"pid": t["Id"], "night_day": day[t.index], "cutoff": t["cutoff"], "y": t["ratio_pct"]})
    return PersonalTask("fitabase", fitabase.TARGET, targets, history, (0.0, 100.0))


def studentlife_task() -> PersonalTask:
    import json

    from . import studentlife as sl

    from .datasets import require_dir

    require_dir(sl.DEFAULT_ROOT / "EMA" / "response" / "Sleep", "StudentLife Sleep EMA folder")
    frames = []
    for f in sorted((sl.DEFAULT_ROOT / "EMA" / "response" / "Sleep").glob("Sleep_u*.json")):
        frames.append(sl.parse_sleep_ema(json.loads(f.read_text()), f.stem.split("_")[1]))
    s = pd.concat(frames, ignore_index=True)
    day = (s["night_date"].dt.tz_localize(None) - pd.Timestamp("1970-01-01")).dt.days
    history = pd.DataFrame({"pid": s["uid"], "night_day": day, "available_at": s["resp_time"], "value": s["hours"]})
    targets = pd.DataFrame({"pid": s["uid"], "night_day": day,
                            "cutoff": s["night_date"] + pd.Timedelta(hours=sl.CUTOFF_HOUR), "y": s["hours"]})
    return PersonalTask("studentlife", sl.TARGET, targets, history, (3.0, 12.0))


# ---------- evaluation ----------

def bootstrap_mae(frame: pd.DataFrame, col: str, ref: str | None = None, n_boot: int = 2000, seed: int = 0) -> dict:
    """Participant bootstrap of MAE (or of the paired MAE difference col - ref)."""
    rng = np.random.default_rng(seed)
    e = frame.assign(a=(frame[col] - frame["y"]).abs())
    if ref:
        e["a"] = e["a"] - (frame[ref] - frame["y"]).abs()
    g = e.groupby("pid")["a"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(c), size=(n_boot, len(c)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    return {"estimate": float(s.sum() / c.sum()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def method_summary(frame: pd.DataFrame, col: str, baseline: str = "expanding_mean") -> dict:
    err = frame[col] - frame["y"]
    per_pid = err.abs().groupby(frame["pid"]).mean()
    out = {
        "mae": float(err.abs().mean()), "rmse": float(np.sqrt((err ** 2).mean())),
        "n_predictions": int(len(frame)), "n_participants": int(frame["pid"].nunique()),
        "mae_ci95": bootstrap_mae(frame, col)["ci95"],
        "participant_mae": {"median": float(per_pid.median()), "q25": float(per_pid.quantile(0.25)),
                            "q75": float(per_pid.quantile(0.75)), "min": float(per_pid.min()), "max": float(per_pid.max())},
    }
    if col != baseline:
        out["paired_vs_expanding_mean"] = bootstrap_mae(frame, col, baseline)
    return out


HISTORY_BINS = [(3, 6), (7, 13), (14, 27), (28, 10_000)]


def by_history_length(frame: pd.DataFrame, methods: list[str]) -> dict:
    out = {}
    for lo, hi in HISTORY_BINS:
        sub = frame[(frame["history_count"] >= lo) & (frame["history_count"] <= hi)]
        if len(sub):
            out[f"{lo}-{hi if hi < 10_000 else 'max'}"] = {
                "n": int(len(sub)), "participants": int(sub["pid"].nunique()),
                **{m: float((sub[m] - sub["y"]).abs().mean()) for m in methods},
            }
    return out


def interval_coverage(frame: pd.DataFrame, center: str = "expanding_mean") -> dict:
    """Empirical coverage of centre +/- z * SD of the person's history (nominal 80 %)."""
    lo = frame[center] - INTERVAL_Z * frame["history_sd"]
    hi = frame[center] + INTERVAL_Z * frame["history_sd"]
    inside = (frame["y"] >= lo) & (frame["y"] <= hi)
    return {
        "nominal": 0.8, "empirical": float(inside.mean()),
        "by_history_length": {
            f"{a}-{b if b < 10_000 else 'max'}": float(inside[(frame["history_count"] >= a) & (frame["history_count"] <= b)].mean())
            for a, b in HISTORY_BINS if ((frame["history_count"] >= a) & (frame["history_count"] <= b)).any()
        },
    }


def select_half_life(train: pd.DataFrame) -> int:
    """Pick the EWMA half-life with the lowest MAE on TRAINING rows only."""
    return min(EWMA_HALF_LIVES, key=lambda h: (train[f"ewma_h{h}"] - train["y"]).abs().mean())


def with_selected_ewma(frame: pd.DataFrame, splits) -> pd.DataFrame:
    """Out-of-sample 'ewma_selected': half-life chosen on each split's training part."""
    frame = frame.copy()
    frame["ewma_selected"] = np.nan
    frame["ewma_selected_half_life"] = np.nan
    for train_mask, test_mask in splits:
        h = select_half_life(frame[train_mask])
        frame.loc[test_mask, "ewma_selected"] = frame.loc[test_mask, f"ewma_h{h}"]
        frame.loc[test_mask, "ewma_selected_half_life"] = h
    return frame


def decision(results_grouped: dict, results_chrono: dict, baseline: str = "expanding_mean") -> dict:
    verdicts = {}
    base = results_grouped[baseline]["mae"]
    for method, g in results_grouped.items():
        if method not in DECISION_CANDIDATES:
            continue
        c = results_chrono[method]
        g_ok = g["paired_vs_expanding_mean"]["ci95"][1] < 0
        c_ok = c["paired_vs_expanding_mean"]["ci95"][1] < 0
        rel = -g["paired_vs_expanding_mean"]["estimate"] / base
        verdicts[method] = {
            "grouped_ci_below_zero": g_ok, "chronological_ci_below_zero": c_ok,
            "relative_improvement_grouped": float(rel),
            "adopt": bool(g_ok and c_ok and rel >= MIN_RELATIVE_IMPROVEMENT),
        }
    adopted = [m for m, v in verdicts.items() if v["adopt"]]
    return {"rule": "CI below zero in both evaluations and >= 2% relative improvement",
            "verdicts": verdicts,
            "selected": min(adopted, key=lambda m: results_grouped[m]["mae"]) if adopted else baseline}
