"""Exploratory MMASH wearable sleep-efficiency experiments.

This script does not modify the original baseline or stacking experiments.
It evaluates fixed, domain-driven feature sets using participant-aware
GroupKFold validation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "sleep_efficiency"
GROUP = "participant_id"

CORE_FEATURES = [
    "heavy_activity_pre_sleep_2h_min",
    "sitting_pre_sleep_4h_min",
    "small_screen_pre_sleep_2h_min",
    "large_screen_pre_sleep_2h_min",
    "vector_mean_pre_sleep_2h",
    "steps_pre_sleep_2h",
    "hr_mean_pre_sleep_2h",
    "hr_std_pre_sleep_2h",
    "laying_pre_sleep_4h_min",
]

HRV_FEATURES = [
    "sdnn_pre_sleep_2h",
    "rmssd_pre_sleep_2h",
    "pnn50_pre_sleep_2h",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df[columns].isna().any().any():
        missing_counts = df[columns].isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(
            "Unexpected missing values in model inputs: "
            f"{missing_counts}"
        )


def evaluate_experiment(
    df: pd.DataFrame,
    experiment_name: str,
    features: list[str],
    models: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate models with participant-level GroupKFold splits."""
    validate_columns(df, [TARGET, GROUP, *features])

    X = df[features]
    y = df[TARGET]
    groups = df[GROUP]

    n_groups = groups.nunique()
    if n_groups < N_SPLITS:
        raise ValueError(
            f"{experiment_name}: {n_groups} unique participants is fewer "
            f"than N_SPLITS={N_SPLITS}."
        )

    splitter = GroupKFold(n_splits=N_SPLITS)
    fold_rows = []
    prediction_rows = []

    for model_name, model in models.items():
        oof_predictions = np.full(len(df), np.nan)

        for fold, (train_index, test_index) in enumerate(
            splitter.split(X, y, groups),
            start=1,
        ):
            train_groups = set(groups.iloc[train_index])
            test_groups = set(groups.iloc[test_index])
            overlap = train_groups.intersection(test_groups)
            if overlap:
                raise RuntimeError(
                    f"Participant leakage in fold {fold}: {sorted(overlap)}"
                )

            fitted_model = clone(model)
            fitted_model.fit(X.iloc[train_index], y.iloc[train_index])
            fold_predictions = fitted_model.predict(X.iloc[test_index])
            oof_predictions[test_index] = fold_predictions

            fold_metrics = regression_metrics(
                y.iloc[test_index],
                fold_predictions,
            )
            fold_rows.append(
                {
                    "experiment": experiment_name,
                    "model": model_name,
                    "fold": fold,
                    "n_train_episodes": len(train_index),
                    "n_test_episodes": len(test_index),
                    "n_train_participants": len(train_groups),
                    "n_test_participants": len(test_groups),
                    **fold_metrics,
                }
            )

        if np.isnan(oof_predictions).any():
            raise RuntimeError(
                f"{experiment_name}/{model_name}: incomplete OOF predictions."
            )

        for row_index, prediction in enumerate(oof_predictions):
            prediction_rows.append(
                {
                    "experiment": experiment_name,
                    "model": model_name,
                    "participant_id": df.iloc[row_index][GROUP],
                    "sleep_episode_id": df.iloc[row_index]["sleep_episode_id"],
                    "actual_sleep_efficiency": y.iloc[row_index],
                    "predicted_sleep_efficiency": prediction,
                    "residual": y.iloc[row_index] - prediction,
                }
            )

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)

    summary_rows = []
    for model_name, model_predictions in predictions.groupby("model"):
        pooled_metrics = regression_metrics(
            model_predictions["actual_sleep_efficiency"],
            model_predictions["predicted_sleep_efficiency"],
        )
        model_fold_metrics = fold_metrics[fold_metrics["model"] == model_name]

        summary_rows.append(
            {
                "experiment": experiment_name,
                "model": model_name,
                "n_episodes": len(df),
                "n_participants": n_groups,
                "n_features": len(features),
                "features": ", ".join(features),
                "pooled_oof_mae": pooled_metrics["mae"],
                "pooled_oof_rmse": pooled_metrics["rmse"],
                "pooled_oof_r2": pooled_metrics["r2"],
                "mean_fold_mae": model_fold_metrics["mae"].mean(),
                "std_fold_mae": model_fold_metrics["mae"].std(),
                "mean_fold_rmse": model_fold_metrics["rmse"].mean(),
                "std_fold_rmse": model_fold_metrics["rmse"].std(),
                "mean_fold_r2": model_fold_metrics["r2"].mean(),
                "std_fold_r2": model_fold_metrics["r2"].std(),
            }
        )

    return fold_metrics, predictions, pd.DataFrame(summary_rows)


def main() -> None:
    root = project_root()
    data_path = (
        root / "ml/data/processed/mmash/mmash_wearable_features_final.csv"
    )
    report_dir = root / "ml/reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    required_quality_columns = [
        "rr_quality_status",
        "rr_hrv_features_available",
    ]
    validate_columns(df, [TARGET, GROUP, "sleep_episode_id"])
    if set(required_quality_columns) - set(df.columns):
        raise ValueError(
            "Dataset is missing HRV quality columns: "
            f"{sorted(set(required_quality_columns) - set(df.columns))}"
        )

    models = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    # Experiment A: all episodes, no HRV-derived predictors.
    core_folds, core_predictions, core_summary = evaluate_experiment(
        df=df,
        experiment_name="core_wearable",
        features=CORE_FEATURES,
        models=models,
    )

    # Experiment B: HRV measurements are considered unavailable when their
    # quality flag is not valid. Those rows are excluded, never zero-filled.
    hrv_usable = (
        df["rr_quality_status"].eq("valid")
        & df["rr_hrv_features_available"].eq(True)
    )
    hrv_df = df.loc[hrv_usable].copy()
    excluded_hrv = df.loc[
        ~hrv_usable,
        [
            "participant_id",
            "sleep_episode_id",
            "rr_quality_status",
            "rr_hrv_features_available",
        ],
    ].copy()

    hrv_folds, hrv_predictions, hrv_summary = evaluate_experiment(
        df=hrv_df,
        experiment_name="core_plus_hrv",
        features=CORE_FEATURES + HRV_FEATURES,
        models=models,
    )

    all_folds = pd.concat([core_folds, hrv_folds], ignore_index=True)
    all_predictions = pd.concat(
        [core_predictions, hrv_predictions],
        ignore_index=True,
    )
    all_summary = pd.concat([core_summary, hrv_summary], ignore_index=True)

    all_folds.to_csv(report_dir / "mmash_wearable_fold_metrics.csv", index=False)
    all_predictions.to_csv(
        report_dir / "mmash_wearable_oof_predictions.csv",
        index=False,
    )
    all_summary.to_csv(
        report_dir / "mmash_wearable_summary_metrics.csv",
        index=False,
    )
    excluded_hrv.to_csv(
        report_dir / "mmash_hrv_excluded_records.csv",
        index=False,
    )

    print("\n=== MMASH WEARABLE EXPERIMENT COMPLETE ===")
    print(f"Dataset: {data_path}")
    print(f"Core experiment: {len(df)} episodes, {df[GROUP].nunique()} participants")
    print(
        "HRV extension: "
        f"{len(hrv_df)} episodes, {hrv_df[GROUP].nunique()} participants"
    )
    print(f"HRV-excluded records: {len(excluded_hrv)}")

    if not excluded_hrv.empty:
        print(excluded_hrv.to_string(index=False))

    print("\n=== POOLED OUT-OF-FOLD METRICS ===")
    print(
        all_summary[
            [
                "experiment",
                "model",
                "pooled_oof_mae",
                "pooled_oof_rmse",
                "pooled_oof_r2",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\nSaved reports:")
    for path in sorted(report_dir.glob("mmash_*.csv")):
        print(f"  {path.relative_to(root)}")


if __name__ == "__main__":
    main()
