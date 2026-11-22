"""Development-only ensemble selection from existing cross-fitted predictions.

This script reads the existing stacking OOF matrix. It does not retrain base
models, alter weights used by the API, write checkpoints, or read the prior
75-row exploratory test results for candidate selection.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[2]
OOF_PATH = ROOT / "ml" / "data" / "processed" / "stacking_oof_predictions.csv"
REPORTS = ROOT / "ml" / "reports"
VERSION = "v1"
RANDOM_STATE = 42
N_SPLITS = 5
BASE_COLUMNS = ["rf_prediction", "xgb_prediction", "hybrid_cnn_prediction"]
TARGET_COLUMN = "sleep_efficiency"
CURRENT_FIXED_WEIGHTS = np.array([0.2, 0.3, 0.5], dtype=float)
EQUAL_WEIGHTS = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
GRID_STEP = 0.05


def metric_summary(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    residual = y_true - prediction
    series = pd.Series(residual)
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "r2": float(r2_score(y_true, prediction)),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "invalid_non_finite": int((~np.isfinite(prediction)).sum()),
        "invalid_outside_0_100": int(((prediction < 0) | (prediction > 100)).sum()),
        "residual_distribution": {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)),
            "median": float(series.median()),
            "q05": float(series.quantile(0.05)),
            "q25": float(series.quantile(0.25)),
            "q75": float(series.quantile(0.75)),
            "q95": float(series.quantile(0.95)),
            "skew": float(series.skew()),
        },
    }


def residual_correlations(y_true: np.ndarray, predictions: dict[str, np.ndarray]) -> dict[str, float]:
    names = list(predictions)
    output: dict[str, float] = {}
    for index, left in enumerate(names):
        left_residual = y_true - predictions[left]
        for right in names[index + 1 :]:
            right_residual = y_true - predictions[right]
            output[f"{left}__{right}"] = float(np.corrcoef(left_residual, right_residual)[0, 1])
    return output


def range_metrics(y_true: np.ndarray, predictions: dict[str, np.ndarray]) -> dict[str, dict]:
    ranges = {
        "low_under_70": y_true < 70,
        "mid_70_to_under_85": (y_true >= 70) & (y_true < 85),
        "high_85_and_above": y_true >= 85,
    }
    result: dict[str, dict] = {}
    for label, mask in ranges.items():
        result[label] = {"n": int(mask.sum()), "models": {}}
        for name, prediction in predictions.items():
            result[label]["models"][name] = {
                "mae": float(mean_absolute_error(y_true[mask], prediction[mask])),
                "rmse": float(np.sqrt(mean_squared_error(y_true[mask], prediction[mask]))),
            }
    return result


def simplex_grid(step: float) -> list[np.ndarray]:
    denominator = round(1 / step)
    weights = []
    for rf in range(denominator + 1):
        for xgb in range(denominator - rf + 1):
            hybrid = denominator - rf - xgb
            weights.append(np.array([rf, xgb, hybrid], dtype=float) / denominator)
    return weights


def select_best_convex_weights(x_train: np.ndarray, y_train: np.ndarray, candidates: list[np.ndarray]) -> np.ndarray:
    # The MAE objective is selected only inside each outer development fold.
    scores = [mean_absolute_error(y_train, x_train @ candidate) for candidate in candidates]
    return candidates[int(np.argmin(scores))]


def nested_meta_predictions(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    linear_prediction = np.empty(len(y), dtype=float)
    convex_prediction = np.empty(len(y), dtype=float)
    linear_folds: list[dict] = []
    convex_folds: list[dict] = []
    candidates = simplex_grid(GRID_STEP)

    for fold, (train_index, validation_index) in enumerate(kfold.split(x), start=1):
        x_train, x_validation = x[train_index], x[validation_index]
        y_train = y[train_index]

        linear = LinearRegression().fit(x_train, y_train)
        linear_prediction[validation_index] = linear.predict(x_validation)
        linear_folds.append({
            "fold": fold,
            "train_rows": int(len(train_index)),
            "validation_rows": int(len(validation_index)),
            "coefficients": [float(value) for value in linear.coef_],
            "intercept": float(linear.intercept_),
        })

        weights = select_best_convex_weights(x_train, y_train, candidates)
        convex_prediction[validation_index] = x_validation @ weights
        convex_folds.append({
            "fold": fold,
            "train_rows": int(len(train_index)),
            "validation_rows": int(len(validation_index)),
            "weights": [float(value) for value in weights],
            "selection_objective": "MAE on the outer-fold training partition only",
        })

    return linear_prediction, convex_prediction, linear_folds, convex_folds


def aggregate_fold_values(folds: list[dict], key: str) -> dict[str, list[float]]:
    values = np.asarray([fold[key] for fold in folds], dtype=float)
    return {"mean": [float(value) for value in values.mean(axis=0)], "std": [float(value) for value in values.std(axis=0, ddof=1)]}


def markdown(report: dict, output: Path) -> None:
    results = report["results"]
    metrics = results["metrics"]
    lines = [
        "# Development-Only Ensemble Selection v1",
        "",
        "This evaluation uses only the existing cross-fitted development predictions. The previously explored 75-row test split was not read or used for model or weight selection.",
        "",
        "## Cross-Validated Development Results",
        "",
        "| Candidate | MAE | RMSE | R2 | Invalid outside 0-100 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in metrics.items():
        lines.append(f"| {name} | {values['mae']:.4f} | {values['rmse']:.4f} | {values['r2']:.4f} | {values['invalid_outside_0_100']} |")
    lines.extend([
        "",
        "## Residual Distribution",
        "",
        "Residuals are calculated as actual sleep efficiency minus predicted sleep efficiency. Values are percentage points.",
        "",
        "| Candidate | Mean | SD | Median | 5th percentile | 95th percentile | Skew |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, values in metrics.items():
        residual = values["residual_distribution"]
        lines.append(
            f"| {name} | {residual['mean']:.4f} | {residual['std']:.4f} | "
            f"{residual['median']:.4f} | {residual['q05']:.4f} | "
            f"{residual['q95']:.4f} | {residual['skew']:.4f} |"
        )

    lines.extend([
        "",
        "## Residual Correlation of Base Models",
        "",
        "Lower correlation indicates more complementary error patterns. This is evidence of potential diversity, not proof that an ensemble improves accuracy.",
        "",
        "| Pair | Residual correlation |",
        "|---|---:|",
    ])
    correlations = results["residual_correlations"]
    for pair in ["rf_oof__xgb_oof", "rf_oof__hybrid_oof", "xgb_oof__hybrid_oof"]:
        lines.append(f"| {pair} | {correlations[pair]:.4f} |")

    lines.extend([
        "",
        "## Performance by Sleep-Efficiency Range",
        "",
        "| Range | Rows | Candidate | MAE | RMSE |",
        "|---|---:|---|---:|---:|",
    ])
    for range_name, values in results["target_range_metrics"].items():
        for candidate in [
            "rf_oof",
            "xgb_oof",
            "hybrid_oof",
            "current_fixed_0_2_0_3_0_5_oof",
            "nested_linear_stacking_oof",
            "nested_convex_blend_oof",
        ]:
            candidate_metrics = values["models"][candidate]
            lines.append(
                f"| {range_name} | {values['n']} | {candidate} | "
                f"{candidate_metrics['mae']:.4f} | {candidate_metrics['rmse']:.4f} |"
            )

    linear = results["nested_linear_coefficient_summary"]
    convex = results["nested_convex_weight_summary"]
    lines.extend([
        "",
        "## Model Contributions in Development Candidates",
        "",
        "| Candidate | RF contribution | XGBoost contribution | Hybrid contribution |",
        "|---|---:|---:|---:|",
        "| Current fixed policy | 0.2000 | 0.3000 | 0.5000 |",
        f"| Nested linear stacking, mean coefficient (SD) | {linear['mean'][0]:.4f} ({linear['std'][0]:.4f}) | {linear['mean'][1]:.4f} ({linear['std'][1]:.4f}) | {linear['mean'][2]:.4f} ({linear['std'][2]:.4f}) |",
        f"| Nested convex blend, mean weight (SD) | {convex['mean'][0]:.4f} ({convex['std'][0]:.4f}) | {convex['mean'][1]:.4f} ({convex['std'][1]:.4f}) | {convex['mean'][2]:.4f} ({convex['std'][2]:.4f}) |",
        "",
        "The nested linear coefficients include an intercept (mean -0.2053; SD 1.5415), so they are not constrained to sum to one. The convex blend is constrained to non-negative weights that sum to one.",
        "",
        "## Selection Boundary",
        "",
        "- RF, XGBoost, Hybrid, current fixed weights, and equal weights use the saved base-model OOF predictions directly.",
        "- Nested linear stacking and nested convex blending are evaluated with a new outer five-fold split over the existing OOF matrix. Base models are not retrained.",
        "- The full-OOF linear fit is reported only as an optimistic diagnostic for the existing stacking approach; it is not a valid selection result.",
        "",
        "## Independent Holdout Assessment",
        "",
        "- No fresh independent holdout remains in the existing small dataset. The original test rows have already been examined, and validation rows participate in existing development/training procedures.",
        "- A future definitive comparison requires a newly locked external cohort or a new participant-level holdout established before retraining any candidate models.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict:
    frame = pd.read_csv(OOF_PATH)
    missing = [column for column in [*BASE_COLUMNS, TARGET_COLUMN] if column not in frame]
    if missing:
        raise ValueError(f"OOF prediction matrix is missing columns: {missing}")
    if frame[BASE_COLUMNS + [TARGET_COLUMN]].isna().any().any():
        raise ValueError("OOF prediction matrix contains missing values.")

    x = frame[BASE_COLUMNS].to_numpy(dtype=float)
    y = frame[TARGET_COLUMN].to_numpy(dtype=float)
    nested_linear, nested_convex, linear_folds, convex_folds = nested_meta_predictions(x, y)

    # Existing full-OOF stacking fit is intentionally marked optimistic because it
    # evaluates a meta-model on the same OOF rows used to train that meta-model.
    full_oof_linear = LinearRegression().fit(x, y).predict(x)
    predictions = {
        "rf_oof": x[:, 0],
        "xgb_oof": x[:, 1],
        "hybrid_oof": x[:, 2],
        "current_fixed_0_2_0_3_0_5_oof": x @ CURRENT_FIXED_WEIGHTS,
        "equal_weight_oof": x @ EQUAL_WEIGHTS,
        "nested_linear_stacking_oof": nested_linear,
        "nested_convex_blend_oof": nested_convex,
        "existing_full_oof_linear_stacking_optimistic": full_oof_linear,
    }
    metrics = {name: metric_summary(y, prediction) for name, prediction in predictions.items()}
    report = {
        "report_version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_development_experiment": True,
        "selection_data": {
            "oof_prediction_matrix": str(OOF_PATH),
            "rows": int(len(frame)),
            "base_prediction_columns": BASE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "target_min": float(y.min()),
            "target_max": float(y.max()),
            "test_split_used_for_selection": False,
        },
        "methodology": {
            "base_models": "Existing RF, XGBoost, and Hybrid out-of-fold predictions stored by the existing stacking experiment.",
            "outer_cross_validation": {"n_splits": N_SPLITS, "shuffle": True, "random_state": RANDOM_STATE},
            "nested_linear_stacking": "LinearRegression fit on each outer-fold training partition of existing OOF base predictions.",
            "nested_convex_blend": {"candidate_weights": "non-negative weights summing to one", "grid_step": GRID_STEP, "selection_objective": "MAE within each outer-fold training partition"},
            "full_oof_linear_note": "Reported as an optimistic diagnostic only because fitting and scoring use the same OOF rows.",
        },
        "results": {
            "metrics": metrics,
            "residual_correlations": residual_correlations(y, predictions),
            "target_range_metrics": range_metrics(y, predictions),
            "nested_linear_fold_coefficients": linear_folds,
            "nested_linear_coefficient_summary": aggregate_fold_values(linear_folds, "coefficients"),
            "nested_linear_intercept_summary": {
                "mean": float(np.mean([fold["intercept"] for fold in linear_folds])),
                "std": float(np.std([fold["intercept"] for fold in linear_folds], ddof=1)),
            },
            "nested_convex_fold_weights": convex_folds,
            "nested_convex_weight_summary": aggregate_fold_values(convex_folds, "weights"),
        },
        "holdout_assessment": {
            "independent_holdout_available": False,
            "reason": "The stored test split has already been examined, the validation partition participates in existing development procedures, and the provided OOF matrix combines stored development rows.",
            "recommended_future_validation": "Lock a new external or participant-level holdout before any approved retraining or final production-policy claim.",
        },
        "limitations": [
            "This experiment reuses existing cross-fitted base predictions; it does not retrain base models.",
            "Nested meta evaluation reduces direct meta-model fitting optimism, but does not replace external validation.",
            "The 75-row historical test split was excluded from selection and cannot be described as untouched final confirmation.",
            "No candidate is authorized for production deployment by this report.",
        ],
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS / f"ensemble_development_selection_{VERSION}.json"
    markdown_path = REPORTS / f"ensemble_development_selection_{VERSION}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown(report, markdown_path)
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps({
        "report": str(REPORTS / f"ensemble_development_selection_{VERSION}.json"),
        "summary": str(REPORTS / f"ensemble_development_selection_{VERSION}.md"),
        "development_rows": report["selection_data"]["rows"],
        "test_split_used_for_selection": report["selection_data"]["test_split_used_for_selection"],
    }, indent=2))
