from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb


def metrics_bundle(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict:
    y_true_cls = (y_true >= threshold).astype(int)
    y_pred_cls = (y_pred >= threshold).astype(int)

    prob = np.clip((y_pred - 50) / 50, 0, 1)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true_cls, y_pred_cls)),
        "precision": float(precision_score(y_true_cls, y_pred_cls, zero_division=0)),
        "recall": float(recall_score(y_true_cls, y_pred_cls, zero_division=0)),
        "f1": float(f1_score(y_true_cls, y_pred_cls, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_cls, prob)),
    }


def run() -> None:
    proc_dir = Path("ml/data/processed")
    art_dir = Path("ml/artifacts")
    rep_dir = Path("ml/reports")
    art_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(proc_dir / "dataset_engineered.csv")
    with (proc_dir / "split.json").open("r", encoding="utf-8") as f:
        split = json.load(f)

    feats = split["feature_columns"]
    threshold = split["quality_threshold"]

    train_df = df.iloc[split["train_idx"]]
    val_df = df.iloc[split["val_idx"]]
    test_df = df.iloc[split["test_idx"]]

    X_train = train_df[feats].values
    y_train = train_df["sleep_efficiency"].values
    X_val = val_df[feats].values
    y_val = val_df["sleep_efficiency"].values
    X_test = test_df[feats].values
    y_test = test_df["sleep_efficiency"].values

    rf = RandomForestRegressor(random_state=42)
    rf_search = RandomizedSearchCV(
        rf,
        {
            "n_estimators": [200, 400, 600],
            "max_depth": [6, 8, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        n_iter=10,
        cv=3,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=1,
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_

    xgbr = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_estimators=600,
        tree_method="hist",
    )
    xgb_search = RandomizedSearchCV(
        xgbr,
        {
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.02, 0.05, 0.08, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "min_child_weight": [1, 3, 5],
        },
        n_iter=12,
        cv=3,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=1,
    )
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_

    rf_pred = best_rf.predict(X_test)
    xgb_pred = best_xgb.predict(X_test)

    rf_metrics = metrics_bundle(y_test, rf_pred, threshold)
    xgb_metrics = metrics_bundle(y_test, xgb_pred, threshold)

    joblib.dump(best_rf, art_dir / "rf_regressor.joblib")
    best_xgb.save_model(art_dir / "xgb_regressor.json")

    bg = train_df[feats].sample(n=min(200, len(train_df)), random_state=42).values.astype(np.float32)
    np.save(art_dir / "shap_background.npy", bg)

    out = {
        "rf": rf_metrics,
        "xgb": xgb_metrics,
        "best_rf_params": rf_search.best_params_,
        "best_xgb_params": xgb_search.best_params_,
        "feature_columns": feats,
        "quality_threshold": threshold,
    }
    with (rep_dir / "tabular_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run()
