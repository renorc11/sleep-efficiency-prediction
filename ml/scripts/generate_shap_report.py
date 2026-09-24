from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb


def run() -> None:
    proc = Path("ml/data/processed")
    art = Path("ml/artifacts")
    rep = Path("ml/reports")
    rep.mkdir(parents=True, exist_ok=True)

    with (proc / "split.json").open("r", encoding="utf-8") as f:
        split = json.load(f)

    feature_cols = split["feature_columns"]
    df = pd.read_csv(proc / "dataset_engineered.csv")
    test_idx = split["test_idx"]
    X_test = df.iloc[test_idx][feature_cols].values

    model = xgb.XGBRegressor()
    model.load_model(str(art / "xgb_regressor.json"))

    background = np.load(art / "shap_background.npy")
    explainer = shap.TreeExplainer(model, background)
    shap_values = explainer.shap_values(X_test[:250])

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[:250], feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(rep / "shap_summary.png", dpi=160)

    print("SHAP summary saved to ml/reports/shap_summary.png")


if __name__ == "__main__":
    run()
