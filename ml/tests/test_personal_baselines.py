"""Tests for personal-history forecasting (ml/forecasting/personal_baselines.py)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.forecasting import personal_baselines as pb  # noqa: E402

BOUNDS = (0.0, 100.0)


def _task(history_rows, target_rows):
    history = pd.DataFrame(history_rows, columns=["pid", "night_day", "available_at", "value"])
    targets = pd.DataFrame(target_rows, columns=["pid", "night_day", "cutoff", "y"])
    return pb.PersonalTask("test", "y", targets, history, BOUNDS)


def test_insufficient_history_gives_no_forecast():
    out = pb.forecast_from_history(np.array([90.0, 80.0]), np.array([1.0, 2.0]), 3.0, BOUNDS)
    assert out == {"history_count": 2}


def test_target_night_future_nights_and_late_reports_are_excluded():
    rows = [(1, d, d + 0.5, 80.0) for d in range(1, 6)]
    rows += [(1, 6, 6.5, 1000.0),   # the target night itself
             (1, 7, 7.5, 1000.0),   # a future night
             (1, 5.5, 6.2, 1000.0)]  # earlier night reported after the cutoff (6.0)
    task = _task(rows, [(1, 6, 6.0, 85.0)])
    f = pb.build_forecasts(task).iloc[0]
    assert f["history_count"] == 5
    assert f["expanding_mean"] == 80.0 and f["ewma_h3"] == 80.0


def test_histories_are_per_participant():
    rows = [(1, d, d, 90.0) for d in range(1, 5)] + [(2, d, d, 50.0) for d in range(1, 5)]
    f = pb.build_forecasts(_task(rows, [(1, 10, 10, 0), (2, 10, 10, 0)])).set_index("pid")
    assert f.loc[1, "expanding_mean"] == 90.0 and f.loc[2, "expanding_mean"] == 50.0


def test_missing_nights_are_skipped_not_zero():
    rows = [(1, 1, 1, 80.0), (1, 2, 2, 90.0), (1, 9, 9, 70.0)]  # nights 3-8 missing
    f = pb.build_forecasts(_task(rows, [(1, 10, 10, 0)])).iloc[0]
    assert f["mean_last7"] == pytest.approx(80.0)
    assert f["history_span_days"] == 9 and f["days_since_last"] == 1


def test_duplicate_history_nights_are_rejected():
    rows = [(1, 1, 1, 80.0), (1, 1, 1, 82.0), (1, 2, 2, 90.0), (1, 3, 3, 85.0)]
    with pytest.raises(AssertionError, match="duplicate"):
        pb.build_forecasts(_task(rows, [(1, 10, 10, 0)]))


def test_windows_and_ewma_weights():
    v = np.arange(1.0, 21.0)
    f = pb.forecast_from_history(v, np.arange(20.0), 20.0, BOUNDS)
    assert f["mean_last7"] == pytest.approx(np.mean(v[-7:]))
    assert f["mean_last14"] == pytest.approx(np.mean(v[-14:]))
    assert f["expanding_mean"] < f["ewma_h14"] < f["ewma_h7"] < f["ewma_h3"] < v[-1]
    assert pb.ewma(np.array([0.0, 10.0]), 1) == pytest.approx(10 / 1.5)


def test_trend_requires_minimum_history_and_is_bounded():
    short = np.linspace(50, 80, 10)
    assert pb.trend_adjusted(short, np.arange(10.0), 10.0, BOUNDS) == pytest.approx(short.mean())
    rising = np.linspace(50, 98, 14)
    days = np.arange(14.0)
    far = pb.trend_adjusted(rising, days, 1000.0, BOUNDS)  # extreme horizon
    assert far == pytest.approx(rising.mean() + rising.std(ddof=1))  # capped at +1 SD
    assert pb.trend_adjusted(np.full(14, 99.0) + np.linspace(0, 0.9, 14), days, 1000.0, BOUNDS) <= 100.0


def test_ewma_half_life_is_selected_on_training_rows_only():
    frame = pd.DataFrame({"y": [0.0, 0.0, 10.0, 10.0], "ewma_h3": [0.0, 0.0, 0.0, 0.0],
                          "ewma_h7": [5.0, 5.0, 10.0, 10.0], "ewma_h14": [9.0, 9.0, 9.0, 9.0]})
    train = np.array([True, True, False, False])
    out = pb.with_selected_ewma(frame, [(train, ~train)])
    assert (out.loc[~train, "ewma_selected_half_life"] == 3).all()  # best on training rows, worst on test


def test_decision_rejects_fixed_half_lives_and_marginal_gains():
    def res(est, hi):
        return {"mae": 1.0 + est, "paired_vs_expanding_mean": {"estimate": est, "ci95": [est - 0.01, hi]}}
    grouped = {"expanding_mean": {"mae": 1.0}, "ewma_h7": res(-0.5, -0.4), "mean_last14": res(-0.01, -0.001),
               "mean_last7": res(-0.1, -0.05), "trend_adjusted": res(-0.1, 0.02), "ewma_selected": res(0.0, 0.1)}
    d = pb.decision(grouped, grouped)
    assert "ewma_h7" not in d["verdicts"]
    assert not d["verdicts"]["mean_last14"]["adopt"]  # 1 % < 2 % threshold
    assert not d["verdicts"]["trend_adjusted"]["adopt"]  # CI crosses zero
    assert d["selected"] == "mean_last7"


def test_bootstrap_resamples_participants():
    frame = pd.DataFrame({"pid": [1, 1, 2, 2], "y": [0.0] * 4, "a": [1.0, 1.0, 3.0, 3.0], "b": [1.0] * 4})
    r = pb.bootstrap_mae(frame, "a", "b", n_boot=500)
    assert r["estimate"] == pytest.approx(1.0)
    assert r["ci95"][0] >= 0.0 and r["ci95"][1] <= 2.0
