from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
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
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class HybridNet(nn.Module):
    def __init__(self, tabular_dim: int):
        super().__init__()
        self.seq_branch = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
        )
        self.tab_branch = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(16 + 128, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x_tab: torch.Tensor, x_seq: torch.Tensor) -> torch.Tensor:
        seq_out = self.seq_branch(x_seq)
        tab_out = self.tab_branch(x_tab)
        return self.head(torch.cat([tab_out, seq_out], dim=1)).squeeze(1)


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


def run(epochs: int = 80, batch_size: int = 64) -> None:
    proc_dir = Path("ml/data/processed")
    art_dir = Path("ml/artifacts")
    rep_dir = Path("ml/reports")
    art_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(proc_dir / "dataset_engineered.csv")
    seq = np.load(proc_dir / "sequences.npy")
    with (proc_dir / "split.json").open("r", encoding="utf-8") as f:
        split = json.load(f)

    feats = split["feature_columns"]
    threshold = split["quality_threshold"]

    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    X_train = df.iloc[train_idx][feats].values.astype(np.float32)
    y_train = df.iloc[train_idx]["sleep_efficiency"].values.astype(np.float32)
    S_train = seq[train_idx].astype(np.float32)

    X_val = df.iloc[val_idx][feats].values.astype(np.float32)
    y_val = df.iloc[val_idx]["sleep_efficiency"].values.astype(np.float32)
    S_val = seq[val_idx].astype(np.float32)

    X_test = df.iloc[test_idx][feats].values.astype(np.float32)
    y_test = df.iloc[test_idx]["sleep_efficiency"].values.astype(np.float32)
    S_test = seq[test_idx].astype(np.float32)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_sc), torch.tensor(S_train[:, None, :]), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    model = HybridNet(tabular_dim=len(feats))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience, wait = 12, 0

    for epoch in range(epochs):
        model.train()
        for xb, sb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb, sb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.tensor(X_val_sc), torch.tensor(S_val[:, None, :])).numpy()
            val_loss = float(np.mean((val_pred - y_val) ** 2))

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_test = model(torch.tensor(X_test_sc), torch.tensor(S_test[:, None, :])).numpy()

    hybrid_metrics = metrics_bundle(y_test, pred_test, threshold)

    torch.save(model.state_dict(), art_dir / "hybrid.pt")
    joblib.dump(scaler, art_dir / "hybrid_tabular_scaler.joblib")

    with (rep_dir / "hybrid_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"hybrid": hybrid_metrics}, f, indent=2)

    print(json.dumps({"hybrid": hybrid_metrics}, indent=2))


if __name__ == "__main__":
    run()
