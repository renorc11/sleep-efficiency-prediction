"""Evaluate personal-history forecasting methods (forecasting_v2).

    backend/.venv/bin/python -m ml.forecasting.evaluate_personal

Writes only to ml/experiments/forecasting_v2/personal_baselines/.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import fitabase, studentlife, walch
from . import personal_baselines as pb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "ml" / "experiments" / "forecasting_v2" / "personal_baselines"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_manifest() -> dict:
    """Content hashes of every raw input file the forecasting code reads."""
    files = []
    for name in ("minuteSleep_merged.csv", "minuteStepsNarrow_merged.csv", "minuteMETsNarrow_merged.csv",
                 "heartrate_seconds_merged.csv"):
        files += glob.glob(str(fitabase.DEFAULT_ROOT / "*" / "*" / name))
    for sub in ("labels", "heart_rate", "steps"):
        files += glob.glob(str(walch.DEFAULT_ROOT / sub / "*.txt"))
    for sub in ("EMA/response/Sleep", "EMA/response/Stress"):
        files += glob.glob(str(studentlife.DEFAULT_ROOT / sub / "*.json"))
    for sub in ("phonelock", "dark", "phonecharge", "activity"):
        files += glob.glob(str(studentlife.DEFAULT_ROOT / "sensing" / sub / "*.csv"))
    home = str(Path.home())
    entries = {p.replace(home, "~"): sha256(p) for p in sorted(files)}
    combined = hashlib.sha256("".join(f"{k}{v}" for k, v in entries.items()).encode()).hexdigest()
    return {"n_files": len(entries), "combined_sha256": combined, "files": entries}


def evaluate(task: pb.PersonalTask) -> dict:
    forecasts = pb.build_forecasts(task)
    total = len(forecasts)
    eligible = forecasts[forecasts["history_count"] >= pb.MIN_HISTORY].reset_index(drop=True)

    # Participant-independent: EWMA half-life chosen on other participants.
    folds = GroupKFold(5).split(eligible, groups=eligible["pid"])
    splits = []
    for tr, te in folds:
        train_mask = eligible.index.isin(tr)
        test_mask = eligible.index.isin(te)
        assert not set(eligible.loc[train_mask, "pid"]) & set(eligible.loc[test_mask, "pid"])
        splits.append((train_mask, test_mask))
    grouped = pb.with_selected_ewma(eligible, splits)

    # Chronological: same date split as forecasting_v1 (70 % quantile of all
    # target nights); half-life chosen on training-period nights only.
    cut = float(np.quantile(task.targets["night_day"], 0.7))
    train_mask = (eligible["night_day"] < cut).to_numpy()
    chrono = pb.with_selected_ewma(eligible, [(train_mask, ~train_mask)])[~train_mask]

    methods = pb.FIXED_METHODS + ["ewma_selected"]
    res_g = {m: pb.method_summary(grouped, m) for m in methods}
    res_c = {m: pb.method_summary(chrono, m) for m in methods}
    hist = eligible["history_count"]
    return {
        "dataset": task.dataset, "target": task.target,
        "target_nights_total": int(total),
        "insufficient_history_nights": int(total - len(eligible)),
        "history_count_per_prediction": {"median": float(hist.median()), "q25": float(hist.quantile(0.25)),
                                         "q75": float(hist.quantile(0.75)), "min": int(hist.min()), "max": int(hist.max())},
        "history_span_days_median": float(eligible["history_span_days"].median()),
        "days_since_last_valid_night": {"median": float(eligible["days_since_last"].median()),
                                        "q95": float(eligible["days_since_last"].quantile(0.95))},
        "trend_applied_share": float(eligible["trend_applied"].mean()),
        "participant_independent": res_g,
        "ewma_half_life_selected_per_fold": sorted(set(grouped["ewma_selected_half_life"].dropna().astype(int))),
        "chronological": {"test_from_day": pd.Timestamp(cut, unit="D").date().isoformat(), "results": res_c,
                          "ewma_half_life_selected": int(chrono["ewma_selected_half_life"].iloc[0])},
        "by_history_length": pb.by_history_length(eligible, methods[:-1]),
        "interval_80_coverage": pb.interval_coverage(eligible),
        "decision": pb.decision(res_g, res_c),
        "_frame": grouped,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report = {
        "version": "forecasting_v2",
        "protocol": {
            "min_history": pb.MIN_HISTORY, "ewma_half_lives": pb.EWMA_HALF_LIVES,
            "trend": {"min_history": pb.TREND_MIN_HISTORY, "window": pb.TREND_WINDOW, "damping": pb.TREND_DAMPING,
                      "adjustment_limit": "+/- 1 SD of window", "fallback": "expanding_mean"},
            "decision_rule": "adopt only if paired participant-bootstrap 95% CI of dMAE vs expanding_mean is < 0 "
                             "in both participant-independent and chronological evaluation and the relative "
                             "improvement is >= 2%",
            "decision_candidates": list(pb.DECISION_CANDIDATES),
            "protocol_correction": "After the first run the fixed EWMA half-lives were removed from the decision "
                                   "candidates (they select a half-life on evaluated data); ewma_selected remains.",
            "walch": "not evaluated: one PSG night per subject, so no personal history exists",
        },
        "dataset_manifest": dataset_manifest(),
        "results": {},
    }
    for task in (pb.fitabase_task(20.0), pb.studentlife_task()):
        r = evaluate(task)
        r.pop("_frame").to_csv(out / f"{task.dataset}_personal_forecasts.csv", index=False)
        report["results"][task.dataset] = r
        print(f"\n== {task.dataset} ({task.target}); eligible {r['target_nights_total'] - r['insufficient_history_nights']}"
              f"/{r['target_nights_total']}, history median {r['history_count_per_prediction']['median']}")
        for label, res in (("participant-independent", r["participant_independent"]),
                           (f"chronological from {r['chronological']['test_from_day']}", r["chronological"]["results"])):
            print(f"  -- {label}")
            for m, s in res.items():
                d = s.get("paired_vs_expanding_mean")
                extra = f"  dMAE {d['estimate']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]" if d else ""
                print(f"  {m:16s} MAE {s['mae']:.3f} [{s['mae_ci95'][0]:.3f}, {s['mae_ci95'][1]:.3f}] RMSE {s['rmse']:.3f} n={s['n_predictions']} p={s['n_participants']}{extra}")
        print("  by history:", json.dumps(r["by_history_length"], default=lambda x: round(x, 3)))
        print("  80% interval coverage:", r["interval_80_coverage"])
        print("  decision:", r["decision"]["selected"])
    (out / "results.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
