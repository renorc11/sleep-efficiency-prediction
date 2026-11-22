"""Read-only comparison of deployed and stacking ensemble artifacts.

This script evaluates only the existing immutable artifacts against the already
stored test indices. It never trains, tunes, saves model weights, or changes
the split, dataset, preprocessing artifacts, or deployed API.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "ml" / "artifacts"
PROCESSED = ROOT / "ml" / "data" / "processed"
REPORTS = ROOT / "ml" / "reports"
VERSION = "v1"
FIXED_WEIGHTS = {"rf": 0.2, "xgb": 0.3, "hybrid": 0.5}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | list[float]]:
    residual = y_true - prediction
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "r2": float(r2_score(y_true, prediction)),
        "prediction_min": float(np.min(prediction)),
        "prediction_max": float(np.max(prediction)),
        "invalid_non_finite": int(np.sum(~np.isfinite(prediction))),
        "invalid_outside_0_100": int(np.sum((prediction < 0) | (prediction > 100))),
        "residuals": residual,
    }


def serializable_metrics(result: dict[str, float | int | list[float]]) -> dict[str, float | int]:
    return {key: value for key, value in result.items() if key != "residuals"}


def residual_correlations(results: dict[str, dict]) -> dict[str, float]:
    labels = list(results)
    output: dict[str, float] = {}
    for index, left in enumerate(labels):
        for right in labels[index + 1 :]:
            left_residuals = results[left]["residuals"]
            right_residuals = results[right]["residuals"]
            output[f"{left}__{right}"] = float(np.corrcoef(left_residuals, right_residuals)[0, 1])
    return output


def target_range_metrics(y_true: np.ndarray, predictions: dict[str, np.ndarray]) -> dict[str, dict]:
    buckets = {
        "low_under_70": y_true < 70,
        "mid_70_to_under_85": (y_true >= 70) & (y_true < 85),
        "high_85_and_above": y_true >= 85,
    }
    output: dict[str, dict] = {}
    for bucket, mask in buckets.items():
        output[bucket] = {"n": int(mask.sum()), "models": {}}
        for model_name, prediction in predictions.items():
            if not mask.any():
                continue
            output[bucket]["models"][model_name] = {
                "mae": float(mean_absolute_error(y_true[mask], prediction[mask])),
                "rmse": float(np.sqrt(mean_squared_error(y_true[mask], prediction[mask]))),
            }
    return output


def load_stacking_module():
    path = ROOT / "ml" / "scripts" / "train_stacking_ensemble.py"
    spec = importlib.util.spec_from_file_location("stacking_training", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load stacking training module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artifact_details(names: list[str]) -> dict[str, dict[str, str | int]]:
    return {
        name: {
            "path": str(ARTIFACTS / name),
            "bytes": (ARTIFACTS / name).stat().st_size,
            "sha256": sha256(ARTIFACTS / name),
        }
        for name in names
    }


def write_markdown(report: dict, path: Path) -> None:
    deployed = report["results"]["deployed_artifacts"]
    stacking = report["results"]["stacking_artifacts"]
    lines = [
        "# Ensemble Artifact Evaluation v1",
        "",
        "This report is read-only. It evaluates existing artifacts against the existing held-out test indices; it does not retrain or tune any model.",
        "",
        "## Held-Out Test Split",
        "",
        f"- Test rows: {report['dataset']['test_rows']}",
        f"- Target range: {report['dataset']['target_min']:.2f}-{report['dataset']['target_max']:.2f}%",
        "",
        "## Comparable Test Results",
        "",
        "| Artifact set / prediction | MAE | RMSE | R2 | Range | Invalid outside 0-100 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group_name, group in [("Deployed", deployed), ("Stacking", stacking)]:
        for model_name, value in group["metrics"].items():
            lines.append(
                f"| {group_name}: {model_name} | {value['mae']:.4f} | {value['rmse']:.4f} | {value['r2']:.4f} | "
                f"{value['prediction_min']:.2f}-{value['prediction_max']:.2f} | {value['invalid_outside_0_100']} |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The deployed fixed-weight ensemble and stacking ensemble use different saved base-model artifacts, so they must not be mixed.",
        "- The stacking meta-model is a linear model over its own out-of-fold-trained artifact set, not a replacement weight vector for the deployed models.",
        "- Output-domain checks are reported separately from model error. No clipping was applied during this evaluation.",
        "",
        "## Limitations",
        "",
        "- This is a single stored test split, not external validation.",
        "- The deployed fixed-weight models and stacking models were trained with different development-data usage and Hybrid CNN definitions.",
        "- No weight tuning, retraining, calibration, or production-path change was performed.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict:
    metadata = json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))
    split = json.loads((PROCESSED / "split.json").read_text(encoding="utf-8"))
    feature_columns = metadata["feature_columns"]
    test_idx = np.asarray(split["test_idx"], dtype=int)
    frame = pd.read_csv(PROCESSED / "dataset_engineered.csv")
    sequences = np.load(PROCESSED / "sequences.npy").astype(np.float32)
    x = frame[feature_columns].to_numpy(dtype=np.float32)
    y = frame["sleep_efficiency"].to_numpy(dtype=np.float32)
    x_test, y_test, sequence_test = x[test_idx], y[test_idx], sequences[test_idx]

    # Deployed API artifacts. These are the artifacts referenced by ModelBundle.
    deployed_rf = joblib.load(ARTIFACTS / "rf_regressor.joblib")
    deployed_xgb = xgb.XGBRegressor()
    deployed_xgb.load_model(str(ARTIFACTS / "xgb_regressor.json"))
    deployed_scaler = joblib.load(ARTIFACTS / "hybrid_tabular_scaler.joblib")
    from app.services.model_service import HybridNet

    deployed_hybrid = HybridNet(tabular_dim=len(feature_columns))
    deployed_hybrid.load_state_dict(torch.load(ARTIFACTS / "hybrid.pt", map_location="cpu"))
    deployed_hybrid.eval()
    with torch.no_grad():
        deployed_hybrid_prediction = deployed_hybrid(
            torch.tensor(deployed_scaler.transform(x_test), dtype=torch.float32),
            torch.tensor(sequence_test[:, None, :], dtype=torch.float32),
        ).numpy()
    deployed_predictions = {
        "rf": deployed_rf.predict(x_test),
        "xgb": deployed_xgb.predict(x_test),
        "hybrid": deployed_hybrid_prediction,
    }
    deployed_predictions["fixed_weight_0_2_0_3_0_5_raw"] = sum(
        FIXED_WEIGHTS[name] * deployed_predictions[name]
        for name in FIXED_WEIGHTS
    )

    # Stacking experiment artifacts. They cannot be combined with deployed ones.
    stacking_module = load_stacking_module()
    stacking_rf = joblib.load(ARTIFACTS / "stacking_final_rf.joblib")
    stacking_xgb = xgb.XGBRegressor()
    stacking_xgb.load_model(str(ARTIFACTS / "stacking_final_xgb.json"))
    stacking_mean = np.load(ARTIFACTS / "stacking_cnn_scaler_mean.npy")
    stacking_std = np.load(ARTIFACTS / "stacking_cnn_scaler_std.npy")
    stacking_hybrid = stacking_module.HybridCNN(len(feature_columns))
    stacking_hybrid.load_state_dict(torch.load(ARTIFACTS / "stacking_final_hybrid_cnn.pt", map_location="cpu"))
    stacking_hybrid.eval()
    with torch.no_grad():
        stacking_hybrid_prediction = stacking_hybrid(
            torch.tensor((x_test - stacking_mean) / stacking_std, dtype=torch.float32),
            torch.tensor(sequence_test, dtype=torch.float32),
        ).numpy()
    stacking_predictions = {
        "rf": stacking_rf.predict(x_test),
        "xgb": stacking_xgb.predict(x_test),
        "hybrid": stacking_hybrid_prediction,
    }
    stacking_predictions["fixed_weight_0_2_0_3_0_5_raw"] = sum(
        FIXED_WEIGHTS[name] * stacking_predictions[name]
        for name in FIXED_WEIGHTS
    )
    stacking_meta = joblib.load(ARTIFACTS / "stacking_meta_model.joblib")
    stacking_predictions["stacking_linear_meta_model"] = stacking_meta.predict(
        np.column_stack([stacking_predictions[name] for name in ("rf", "xgb", "hybrid")])
    )

    deployed_results = {name: metrics(y_test, value) for name, value in deployed_predictions.items()}
    stacking_results = {name: metrics(y_test, value) for name, value in stacking_predictions.items()}
    report = {
        "report_version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_evaluation": True,
        "dataset": {
            "dataset_path": str(PROCESSED / "dataset_engineered.csv"),
            "sequence_path": str(PROCESSED / "sequences.npy"),
            "split_path": str(PROCESSED / "split.json"),
            "test_rows": int(len(test_idx)),
            "target_min": float(y_test.min()),
            "target_max": float(y_test.max()),
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
        },
        "deployed_artifact_set": {
            "aggregation": "0.2 * rf + 0.3 * xgb + 0.5 * hybrid; evaluated without production clipping",
            "hybrid_input": "StandardScaler-transformed tabular features plus sequence shaped [n, 1, 288]",
            "artifacts": artifact_details(["rf_regressor.joblib", "xgb_regressor.json", "hybrid.pt", "hybrid_tabular_scaler.joblib"]),
        },
        "stacking_artifact_set": {
            "aggregation": "LinearRegression meta-model over RF, XGBoost, and Hybrid predictions",
            "hybrid_input": "numpy mean/std standardized tabular features plus sequence shaped [n, 288]",
            "meta_coefficients": [float(value) for value in stacking_meta.coef_],
            "meta_intercept": float(stacking_meta.intercept_),
            "artifacts": artifact_details([
                "stacking_final_rf.joblib", "stacking_final_xgb.json", "stacking_final_hybrid_cnn.pt",
                "stacking_cnn_scaler_mean.npy", "stacking_cnn_scaler_std.npy", "stacking_meta_model.joblib",
            ]),
        },
        "results": {
            "deployed_artifacts": {
                "metrics": {name: serializable_metrics(value) for name, value in deployed_results.items()},
                "residual_correlations": residual_correlations(deployed_results),
                "target_range_metrics": target_range_metrics(y_test, deployed_predictions),
            },
            "stacking_artifacts": {
                "metrics": {name: serializable_metrics(value) for name, value in stacking_results.items()},
                "residual_correlations": residual_correlations(stacking_results),
                "target_range_metrics": target_range_metrics(y_test, stacking_predictions),
            },
        },
        "compatibility_limitations": [
            "The deployed and stacking artifact sets were trained separately and must not be mixed.",
            "The stacking RF and XGBoost final models use train-plus-validation development data, whereas deployed RF and XGBoost artifacts use the baseline training path.",
            "The deployed HybridNet and stacking HybridCNN use different module layouts and sequence-convolution padding behavior.",
            "Both Hybrid output heads are unconstrained linear regressors; no clipping was applied in this evaluation.",
            "The held-out split is internal validation only and does not establish external generalization.",
        ],
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS / f"ensemble_artifact_evaluation_{VERSION}.json"
    markdown_path = REPORTS / f"ensemble_artifact_evaluation_{VERSION}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, markdown_path)
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "report": str(REPORTS / f"ensemble_artifact_evaluation_{VERSION}.json"),
        "markdown_summary": str(REPORTS / f"ensemble_artifact_evaluation_{VERSION}.md"),
        "test_rows": result["dataset"]["test_rows"],
    }, indent=2))
