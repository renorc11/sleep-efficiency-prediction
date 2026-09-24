from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold
from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5

DATA_PATH = Path("ml/data/processed/dataset_engineered.csv")
SPLIT_PATH = Path("ml/data/processed/split.json")

ARTIFACT_DIR = Path("ml/artifacts")
REPORT_DIR = Path("ml/reports")
OUTPUT_DIR = Path("ml/data/processed")

TARGET = "sleep_efficiency"

# The original baseline contains exactly these 20 predictors.
FEATURE_COLUMNS = [
    "caffeine_mg_evening",
    "alcohol_units",
    "dinner_hour",
    "stress_level",
    "mood_score",
    "physical_activity_min",
    "naps_count",
    "naps_total_min",
    "previous_sleep_hours",
    "work_hours",
    "blue_light_minutes",
    "bedtime_consistency",
    "hydration_liters",
    "inferred_sleep_start_hour",
    "longest_evening_inactivity_min",
    "nighttime_activity_bursts",
    "dinner_to_sleep_gap",
    "caffeine_stress_interaction",
    "activity_stress_balance",
    "nap_sleep_debt_proxy",
]

SEQUENCE_PATH = Path("ml/data/processed/sequences.npy")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# HYBRID CNN
# ============================================================

class HybridCNN(nn.Module):
    def __init__(self, n_tabular_features: int):
        super().__init__()

        self.tabular_branch = nn.Sequential(
            nn.Linear(n_tabular_features, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.sequence_branch = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=5),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )

        self.head = nn.Sequential(
            nn.Linear(16 * 8 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        tabular: torch.Tensor,
        sequence: torch.Tensor,
    ) -> torch.Tensor:

        tabular_features = self.tabular_branch(tabular)

        sequence = sequence.unsqueeze(1)
        sequence_features = self.sequence_branch(sequence)
        sequence_features = sequence_features.flatten(start_dim=1)

        combined = torch.cat(
            [tabular_features, sequence_features],
            dim=1,
        )

        return self.head(combined).squeeze(1)


# ============================================================
# METRICS
# ============================================================

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "r2": float(r2_score(y_true, y_pred)),
    }


def threshold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 85.0,
) -> dict:

    true_label = (y_true >= threshold).astype(int)
    pred_label = (y_pred >= threshold).astype(int)

    tp = int(np.sum((true_label == 1) & (pred_label == 1)))
    tn = int(np.sum((true_label == 0) & (pred_label == 0)))
    fp = int(np.sum((true_label == 0) & (pred_label == 1)))
    fn = int(np.sum((true_label == 1) & (pred_label == 0)))

    accuracy = (tp + tn) / len(y_true)

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


# ============================================================
# TABULAR MODELS
# ============================================================

def make_rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=600,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_xgb() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_estimators=600,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=1.0,
        min_child_weight=1,
        tree_method="hist",
    )


# ============================================================
# HYBRID CNN TRAINING
# ============================================================

def train_cnn(
    X_tab_train: np.ndarray,
    X_seq_train: np.ndarray,
    y_train: np.ndarray,
    X_tab_val: np.ndarray,
    X_seq_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 80,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 12,
) -> HybridCNN:

    set_seed(RANDOM_STATE)

    model = HybridCNN(
        n_tabular_features=X_tab_train.shape[1]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.MSELoss()

    X_tab_train_t = torch.tensor(
        X_tab_train,
        dtype=torch.float32,
    )

    X_seq_train_t = torch.tensor(
        X_seq_train,
        dtype=torch.float32,
    )

    y_train_t = torch.tensor(
        y_train,
        dtype=torch.float32,
    )

    X_tab_val_t = torch.tensor(
        X_tab_val,
        dtype=torch.float32,
    )

    X_seq_val_t = torch.tensor(
        X_seq_val,
        dtype=torch.float32,
    )

    y_val_t = torch.tensor(
        y_val,
        dtype=torch.float32,
    )

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0

    n_train = len(y_train)

    for epoch in range(epochs):

        model.train()

        permutation = torch.randperm(n_train)

        for start in range(0, n_train, batch_size):

            idx = permutation[start:start + batch_size]

            batch_tab = X_tab_train_t[idx]
            batch_seq = X_seq_train_t[idx]
            batch_y = y_train_t[idx]

            optimizer.zero_grad()

            prediction = model(
                batch_tab,
                batch_seq,
            )

            loss = criterion(
                prediction,
                batch_y,
            )

            loss.backward()
            optimizer.step()

        # -----------------------------
        # Validation
        # -----------------------------

        model.eval()

        with torch.no_grad():

            val_prediction = model(
                X_tab_val_t,
                X_seq_val_t,
            )

            val_loss = criterion(
                val_prediction,
                y_val_t,
            ).item()

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def cnn_predict(
    model: HybridCNN,
    X_tab: np.ndarray,
    X_seq: np.ndarray,
) -> np.ndarray:

    model.eval()

    X_tab_t = torch.tensor(
        X_tab,
        dtype=torch.float32,
    )

    X_seq_t = torch.tensor(
        X_seq,
        dtype=torch.float32,
    )

    with torch.no_grad():

        prediction = model(
            X_tab_t,
            X_seq_t,
        ).cpu().numpy()

    return prediction


# ============================================================
# MAIN STACKING PIPELINE
# ============================================================

def main() -> None:

    set_seed(RANDOM_STATE)

    print("=" * 80)
    print("LEAKAGE-SAFE STACKING ENSEMBLE")
    print("=" * 80)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    sequences = np.load(SEQUENCE_PATH)

    with SPLIT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        split = json.load(f)

    train_idx = np.array(
        split["train_idx"],
        dtype=int,
    )

    val_idx = np.array(
        split["val_idx"],
        dtype=int,
    )

    test_idx = np.array(
        split["test_idx"],
        dtype=int,
    )

    print(f"Dataset shape: {df.shape}")
    print(f"Train: {len(train_idx)}")
    print(f"Validation: {len(val_idx)}")
    print(f"Test: {len(test_idx)}")

    # --------------------------------------------------------
    # Verify feature schema
    # --------------------------------------------------------

    actual_features = split["feature_columns"]

    if actual_features != FEATURE_COLUMNS:
        raise ValueError(
            "Feature schema in split.json does not match "
            "the expected 20-feature baseline schema."
        )

    X = df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32
    )

    y = df[TARGET].to_numpy(
        dtype=np.float32
    )

    # --------------------------------------------------------
    # We combine original train + validation for OOF
    # generation.
    #
    # The original 75-sample test set remains untouched.
    # --------------------------------------------------------

    development_idx = np.concatenate(
        [train_idx, val_idx]
    )

    test_indices = test_idx

    X_dev = X[development_idx]
    y_dev = y[development_idx]

    seq_dev = sequences[development_idx]

    X_test = X[test_indices]
    y_test = y[test_indices]

    seq_test = sequences[test_indices]

    print(
        f"Development samples for OOF stacking: "
        f"{len(development_idx)}"
    )

    print(
        f"Final untouched test samples: "
        f"{len(test_indices)}"
    )

    # --------------------------------------------------------
    # OOF arrays
    # --------------------------------------------------------

    oof_rf = np.zeros(len(development_idx))
    oof_xgb = np.zeros(len(development_idx))
    oof_cnn = np.zeros(len(development_idx))

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # OOF training
    # --------------------------------------------------------

    for fold, (fold_train, fold_val) in enumerate(
        kfold.split(X_dev),
        start=1,
    ):

        print()
        print("-" * 80)
        print(f"FOLD {fold}/{N_SPLITS}")
        print("-" * 80)

        X_train = X_dev[fold_train]
        X_val = X_dev[fold_val]

        y_train = y_dev[fold_train]
        y_val = y_dev[fold_val]

        seq_train = seq_dev[fold_train]
        seq_val = seq_dev[fold_val]

        # ====================================================
        # RANDOM FOREST
        # ====================================================

        print("Training Random Forest...")

        rf = make_rf()

        rf.fit(
            X_train,
            y_train,
        )

        oof_rf[fold_val] = rf.predict(X_val)

        # ====================================================
        # XGBOOST
        # ====================================================

        print("Training XGBoost...")

        xgb = make_xgb()

        xgb.fit(
            X_train,
            y_train,
        )

        oof_xgb[fold_val] = xgb.predict(X_val)

        # ====================================================
        # HYBRID CNN
        # ====================================================

        print("Training Hybrid CNN...")

        # Fold-specific standardization.
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)

        std[std == 0] = 1.0

        X_train_scaled = (
            X_train - mean
        ) / std

        X_val_scaled = (
            X_val - mean
        ) / std

        cnn = train_cnn(
            X_train_scaled,
            seq_train,
            y_train,
            X_val_scaled,
            seq_val,
            y_val,
        )

        oof_cnn[fold_val] = cnn_predict(
            cnn,
            X_val_scaled,
            seq_val,
        )

        print(
            "Fold complete."
        )

    # --------------------------------------------------------
    # Build meta-training matrix
    # --------------------------------------------------------

    meta_X = np.column_stack(
        [
            oof_rf,
            oof_xgb,
            oof_cnn,
        ]
    )

    meta_columns = [
        "rf_prediction",
        "xgb_prediction",
        "hybrid_cnn_prediction",
    ]

    meta_df = pd.DataFrame(
        meta_X,
        columns=meta_columns,
    )

    meta_df[TARGET] = y_dev

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    meta_df.to_csv(
        OUTPUT_DIR / "stacking_oof_predictions.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("OOF PREDICTIONS CREATED")
    print("=" * 80)

    # --------------------------------------------------------
    # Train meta learner
    # --------------------------------------------------------

    print("Training Linear Regression meta learner...")

    meta_model = LinearRegression()

    meta_model.fit(
        meta_X,
        y_dev,
    )

    print(
        "Meta-model coefficients:",
        meta_model.coef_,
    )

    print(
        "Meta-model intercept:",
        meta_model.intercept_,
    )

    # --------------------------------------------------------
    # Train final base models on entire development set
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("TRAINING FINAL BASE MODELS")
    print("=" * 80)

    final_rf = make_rf()

    final_rf.fit(
        X_dev,
        y_dev,
    )

    final_xgb = make_xgb()

    final_xgb.fit(
        X_dev,
        y_dev,
    )

    # --------------------------------------------------------
    # Final CNN
    #
    # Use the original validation portion as the CNN's
    # early-stopping validation set while keeping the test
    # set untouched.
    # --------------------------------------------------------

    cnn_scaler_mean = X[train_idx].mean(axis=0)
    cnn_scaler_std = X[train_idx].std(axis=0)

    cnn_scaler_std[
        cnn_scaler_std == 0
    ] = 1.0

    X_train_scaled = (
        X[train_idx] - cnn_scaler_mean
    ) / cnn_scaler_std

    X_val_scaled = (
        X[val_idx] - cnn_scaler_mean
    ) / cnn_scaler_std

    final_cnn = train_cnn(
        X_train_scaled,
        sequences[train_idx],
        y[train_idx],
        X_val_scaled,
        sequences[val_idx],
        y[val_idx],
    )

    # --------------------------------------------------------
    # Generate test predictions
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("TEST PREDICTIONS")
    print("=" * 80)

    test_rf_pred = final_rf.predict(
        X_test
    )

    test_xgb_pred = final_xgb.predict(
        X_test
    )

    X_test_scaled = (
        X_test - cnn_scaler_mean
    ) / cnn_scaler_std

    test_cnn_pred = cnn_predict(
        final_cnn,
        X_test_scaled,
        seq_test,
    )

    test_meta_X = np.column_stack(
        [
            test_rf_pred,
            test_xgb_pred,
            test_cnn_pred,
        ]
    )

    stacking_pred = meta_model.predict(
        test_meta_X
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    results = {}

    results["random_forest"] = {
        **regression_metrics(
            y_test,
            test_rf_pred,
        ),
        **threshold_metrics(
            y_test,
            test_rf_pred,
        ),
    }

    results["xgboost"] = {
        **regression_metrics(
            y_test,
            test_xgb_pred,
        ),
        **threshold_metrics(
            y_test,
            test_xgb_pred,
        ),
    }

    results["hybrid_cnn"] = {
        **regression_metrics(
            y_test,
            test_cnn_pred,
        ),
        **threshold_metrics(
            y_test,
            test_cnn_pred,
        ),
    }

    results["stacking"] = {
        **regression_metrics(
            y_test,
            stacking_pred,
        ),
        **threshold_metrics(
            y_test,
            stacking_pred,
        ),
    }

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()

    for model_name, metrics in results.items():

        print(
            f"{model_name:20s} "
            f"MAE={metrics['mae']:.4f} "
            f"RMSE={metrics['rmse']:.4f} "
            f"R2={metrics['r2']:.4f}"
        )

    # --------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        meta_model,
        ARTIFACT_DIR / "stacking_meta_model.joblib",
    )

    joblib.dump(
        final_rf,
        ARTIFACT_DIR / "stacking_final_rf.joblib",
    )

    final_xgb.save_model(
        ARTIFACT_DIR / "stacking_final_xgb.json"
    )

    torch.save(
        final_cnn.state_dict(),
        ARTIFACT_DIR / "stacking_final_hybrid_cnn.pt",
    )

    np.save(
        ARTIFACT_DIR / "stacking_cnn_scaler_mean.npy",
        cnn_scaler_mean,
    )

    np.save(
        ARTIFACT_DIR / "stacking_cnn_scaler_std.npy",
        cnn_scaler_std,
    )

    report = {
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "development_size": int(len(development_idx)),
        "test_size": int(len(test_indices)),
        "feature_columns": FEATURE_COLUMNS,
        "meta_features": meta_columns,
        "meta_model": "LinearRegression",
        "results": results,
        "meta_coefficients": [
            float(x)
            for x in meta_model.coef_
        ],
        "meta_intercept": float(
            meta_model.intercept_
        ),
    }

    with (
        REPORT_DIR
        / "stacking_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("STACKING COMPLETE")
    print("=" * 80)

    print(
        "Saved OOF predictions to:",
        OUTPUT_DIR / "stacking_oof_predictions.csv",
    )

    print(
        "Saved metrics to:",
        REPORT_DIR / "stacking_metrics.json",
    )


if __name__ == "__main__":
    main()