"""Run every forecasting task and the legacy-model re-evaluation.

Usage (from the project root):
    backend/.venv/bin/python -m ml.forecasting.evaluate [--out ml/experiments/forecasting_v1]

Outputs are written only to the experiment directory. No deployed artifact,
dataset, database or API file is read for writing or modified.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, LeaveOneOut

from . import FORECASTING_VERSION, fitabase, studentlife, walch
from .common import evaluate_grouped, make_models, regression_metrics, summarise

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORY_ONLY = {
    "fitabase": ["prev_available", "prev_minutes", "prev_asleep_min", "prev_ratio_pct", "prev_awake_min",
                 "prev_restless_min", "prev_onset_hour", "prev_wake_hour", "hist_mean", "hist_mean_last7",
                 "hist_count", "nights_observed_last7"],
    "studentlife": ["prev_available", "prev_sleep_hours", "prev_sleep_rate", "hist_mean", "hist_mean_last7", "hist_count"],
}


def grouped_splits(frame, group, n_splits=5):
    return list(GroupKFold(n_splits=n_splits).split(frame, groups=frame[group]))


def chronological_splits(frame, fraction=0.7):
    """One split by calendar date: training nights strictly before test nights."""
    cut = np.quantile(frame["night_order"], fraction)
    tr = np.where(frame["night_order"] < cut)[0]
    te = np.where(frame["night_order"] >= cut)[0]
    return [(tr, te)], int(cut)


def add_history_only_model(frame, oof, features, target, splits):
    """Random forest restricted to sleep-history features (ablation)."""
    from sklearn.ensemble import RandomForestRegressor
    from .common import RF_PARAMS

    frame = frame.reset_index(drop=True)
    for tr, te in splits:
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(frame.iloc[tr][features].to_numpy(float), frame.iloc[tr][target].to_numpy(float))
        oof.loc[te, "random_forest_history_only"] = model.predict(frame.iloc[te][features].to_numpy(float))
    return oof


def run_task(name, dataset, frame, features, target, group, splits, carry, hist, validation, notes, out_dir, history_only=None):
    frame = frame.reset_index(drop=True)
    oof = evaluate_grouped(frame, features, target, group, splits, carry, hist)
    if history_only:
        oof = add_history_only_model(frame, oof, history_only, target, splits)
    result = summarise(oof, target, group, name, dataset, validation, notes)
    if history_only:
        result.metrics["random_forest_history_only"] = regression_metrics(oof[target], oof["random_forest_history_only"])
    oof.to_csv(out_dir / f"{name}_predictions.csv", index=False)
    return result


def feature_importance(frame, features, target):
    """Impurity importance of a random forest fitted on all rows (descriptive only)."""
    model = make_models()["random_forest"]()
    model.fit(frame[features].to_numpy(float), frame[target].to_numpy(float))
    return dict(sorted(zip(features, map(float, model.feature_importances_)), key=lambda kv: -kv[1]))


def legacy_reevaluation() -> dict:
    """Re-evaluate the deployed model family on its own development data.

    The deployed split is a random row split. The 374 rows come from only 132
    distinct source profiles, so identical profiles appear in train and test.
    This refits the same hyperparameters with (a) random 5-fold CV and (b)
    5-fold CV grouped by source profile. Deployed artifacts are not touched.
    """
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor

    df = pd.read_csv(PROJECT_ROOT / "ml/data/processed/dataset_engineered.csv")
    src = pd.read_csv(PROJECT_ROOT / "ml/data/raw/source_notebooks/Sleep_health_and_lifestyle_dataset.csv")
    split = json.loads((PROJECT_ROOT / "ml/data/processed/split.json").read_text())
    params = json.loads((PROJECT_ROOT / "ml/reports/tabular_metrics.json").read_text())
    feats = split["feature_columns"]
    profile = pd.factorize(src.drop(columns=["Person ID"]).astype(str).agg("|".join, axis=1))[0]
    X, y = df[feats].to_numpy(float), df["sleep_efficiency"].to_numpy(float)
    test_profiles_seen = int(np.isin(profile[split["test_idx"]], profile[split["train_idx"] + split["val_idx"]]).sum())

    factories = {
        "random_forest": lambda: RandomForestRegressor(random_state=42, n_jobs=1, **params["best_rf_params"]),
        "xgboost": lambda: xgb.XGBRegressor(objective="reg:squarederror", n_estimators=600, tree_method="hist",
                                            random_state=42, **params["best_xgb_params"]),
    }
    results = {}
    for scheme, splits in (
        ("random_row_5fold", list(KFold(5, shuffle=True, random_state=42).split(X))),
        ("profile_grouped_5fold", list(GroupKFold(5).split(X, groups=profile))),
    ):
        for name, factory in factories.items():
            pred = np.zeros_like(y)
            for tr, te in splits:
                pred[te] = factory().fit(X[tr], y[tr]).predict(X[te])
            results[f"{scheme}/{name}"] = regression_metrics(y, pred)
        mean_pred = np.zeros_like(y)
        for tr, te in splits:
            mean_pred[te] = y[tr].mean()
        results[f"{scheme}/baseline_training_mean"] = regression_metrics(y, mean_pred)
    return {
        "rows": int(len(df)), "distinct_source_profiles": int(profile.max() + 1),
        "deployed_test_rows": len(split["test_idx"]),
        "deployed_test_rows_with_identical_profile_in_train_or_val": test_profiles_seen,
        "corr_previous_sleep_hours_vs_same_night_duration": float(np.corrcoef(df["previous_sleep_hours"], src["Sleep Duration"])[0, 1]),
        "results": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(PROJECT_ROOT / "ml" / "experiments" / FORECASTING_VERSION))
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results, tables = [], {}

    # Fitabase: primary 20:00 cutoff, 21:00 sensitivity.
    for cutoff in (20.0, 21.0):
        fb = fitabase.build_feature_table(cutoff)
        tag = f"{int(cutoff)}00"
        tables[f"fitabase_{tag}"] = fb
        fb.to_csv(out_dir / f"fitabase_features_cutoff{tag}.csv", index=False)
        note = [f"cutoff {int(cutoff)}:00 local; target = device sleep ratio proxy, not PSG efficiency"]
        results.append(run_task(
            f"fitabase_cutoff{tag}_participant_grouped", "fitabase", fb, fitabase.FEATURES, fitabase.TARGET, "Id",
            grouped_splits(fb, "Id"), "prev_ratio_pct", "hist_mean", "GroupKFold(5) by participant", note, out_dir,
            HISTORY_ONLY["fitabase"]))
        if cutoff == 20.0:
            splits, cut = chronological_splits(fb)
            results.append(run_task(
                "fitabase_cutoff2000_chronological", "fitabase", fb, fitabase.FEATURES, fitabase.TARGET, "Id",
                splits, "prev_ratio_pct", "hist_mean",
                f"single chronological split: train nights before {pd.Timestamp(cut, unit='D').date()}, test on/after",
                note + ["participants appear in both periods; no test night precedes a training night"], out_dir,
                HISTORY_ONLY["fitabase"]))

    wa = walch.build_feature_table()
    tables["walch"] = wa
    wa.to_csv(out_dir / "walch_features.csv", index=False)
    results.append(run_task(
        "walch_leave_one_subject_out", "walch", wa, walch.FEATURES, walch.TARGET, "subject",
        list(LeaveOneOut().split(wa)), None, None, "leave-one-subject-out (31 folds)",
        ["one PSG night per subject; target = PSG sleep efficiency; cutoff 30 min before PSG start"], out_dir))

    sl = studentlife.build_feature_table()
    tables["studentlife"] = sl
    sl.to_csv(out_dir / "studentlife_features.csv", index=False)
    note = ["target = self-reported sleep hours (not efficiency); cutoff 21:00 America/New_York (assumed)"]
    results.append(run_task(
        "studentlife_participant_grouped", "studentlife", sl, studentlife.FEATURES, studentlife.TARGET, "uid",
        grouped_splits(sl, "uid"), "prev_sleep_hours", "hist_mean", "GroupKFold(5) by participant", note, out_dir,
        HISTORY_ONLY["studentlife"]))
    splits, cut = chronological_splits(sl)
    results.append(run_task(
        "studentlife_chronological", "studentlife", sl, studentlife.FEATURES, studentlife.TARGET, "uid",
        splits, "prev_sleep_hours", "hist_mean",
        f"single chronological split: train nights before {pd.Timestamp(cut, unit='D').date()}, test on/after",
        note, out_dir, HISTORY_ONLY["studentlife"]))

    importances = {
        "fitabase_2000": feature_importance(tables["fitabase_2000"], fitabase.FEATURES, fitabase.TARGET),
        "studentlife": feature_importance(sl, studentlife.FEATURES, studentlife.TARGET),
    }
    report = {
        "version": FORECASTING_VERSION,
        "protocol_notes": [
            "Hyperparameters are fixed in common.py and were not tuned on any evaluation fold.",
            "The *_residual variants (forecast deviation from personal history) were added after the first "
            "run showed direct models underperforming the personal-history baseline; both are reported.",
            "Feature importances are fitted on all rows and are descriptive only, not validation evidence.",
        ],
        "tasks": [asdict(r) for r in results],
        "feature_importance_descriptive": importances,
        "legacy_reevaluation": legacy_reevaluation(),
        "rf_params": __import__("ml.forecasting.common", fromlist=["RF_PARAMS"]).RF_PARAMS,
        "xgb_params": __import__("ml.forecasting.common", fromlist=["XGB_PARAMS"]).XGB_PARAMS,
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2, default=str))
    for r in results:
        print(f"\n== {r.task} ({r.n_participants} participants, {r.n_nights} nights; {r.validation})")
        for k, m in r.metrics.items():
            print(f"  {k:32s} MAE {m['mae']:.3f}  RMSE {m['rmse']:.3f}  R2 {m['r2']:.3f}")
        for k, p in r.paired_vs_best_baseline.items():
            print(f"  {k} vs {p['baseline']}: dMAE {p['mae_difference']:+.3f} CI95 [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]")
    print("\n== legacy re-evaluation")
    print(json.dumps(report["legacy_reevaluation"], indent=2))


if __name__ == "__main__":
    main()
