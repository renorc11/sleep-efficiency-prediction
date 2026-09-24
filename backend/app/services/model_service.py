from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import shap
import torch
import xgboost as xgb
from torch import nn

from app.core.config import settings

FEATURE_LABELS = {
    "caffeine_mg_evening": "Evening caffeine intake",
    "alcohol_units": "Alcohol intake",
    "dinner_hour": "Dinner timing",
    "stress_level": "Stress level",
    "mood_score": "Mood",
    "physical_activity_min": "Physical activity",
    "naps_count": "Number of naps",
    "naps_total_min": "Total nap duration",
    "previous_sleep_hours": "Previous night's sleep duration",
    "work_hours": "Work duration",
    "blue_light_minutes": "Blue-light exposure",
    "bedtime_consistency": "Bedtime consistency",
    "hydration_liters": "Hydration",
    "inferred_sleep_start_hour": "Estimated sleep start time",
    "longest_evening_inactivity_min": "Longest evening inactivity period",
    "nighttime_activity_bursts": "Nighttime phone checks",
    "dinner_to_sleep_gap": "Dinner-to-sleep gap",
    "caffeine_stress_interaction": "Combined caffeine + stress load",
    "activity_stress_balance": "Activity-to-stress balance",
    "nap_sleep_debt_proxy": "Recovery pressure from sleep debt",
}

RISK_HIGHER_IS_WORSE = {
    "caffeine_mg_evening",
    "alcohol_units",
    "stress_level",
    "blue_light_minutes",
    "nighttime_activity_bursts",
    "caffeine_stress_interaction",
    "nap_sleep_debt_proxy",
}

PROTECTIVE_HIGHER_IS_BETTER = {
    "physical_activity_min",
    "previous_sleep_hours",
    "bedtime_consistency",
    "hydration_liters",
    "activity_stress_balance",
}


def build_human_insight(feature_key: str, feature_name: str, impact: float) -> str:
    if feature_key in RISK_HIGHER_IS_WORSE:
        if impact >= 0:
            return f"{feature_name} is generally a risk factor, but your current level appears relatively controlled."
        return f"{feature_name} is likely acting as a risk factor right now and may be lowering sleep quality."

    if feature_key in PROTECTIVE_HIGHER_IS_BETTER:
        if impact >= 0:
            return f"{feature_name} is currently supporting better sleep quality."
        return f"{feature_name} appears lower than ideal and may be limiting sleep quality."

    if impact >= 0:
        return f"{feature_name} is currently contributing positively to this prediction."
    return f"{feature_name} is currently contributing negatively to this prediction."


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
        self.head = nn.Sequential(
            nn.Linear(16 + 16 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_tab: torch.Tensor, x_seq: torch.Tensor) -> torch.Tensor:
        seq_out = self.seq_branch(x_seq)
        tab_out = self.tab_branch(x_tab)
        return self.head(torch.cat([tab_out, seq_out], dim=1)).squeeze(1)


class ModelBundle:
    def __init__(self) -> None:
        artifacts = settings.artifacts_dir
        if not artifacts.exists():
            raise FileNotFoundError(f"Artifacts directory missing: {artifacts}")

        with (artifacts / "metadata.json").open("r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.features = self.meta["feature_columns"]
        self.quality_threshold = float(self.meta["quality_threshold"])

        self.rf = joblib.load(artifacts / "rf_regressor.joblib")

        self.xgb = xgb.XGBRegressor()
        self.xgb.load_model(str(artifacts / "xgb_regressor.json"))

        self.hybrid_scaler = joblib.load(artifacts / "hybrid_tabular_scaler.joblib")
        self.hybrid = HybridNet(tabular_dim=len(self.features))
        self.hybrid.load_state_dict(torch.load(artifacts / "hybrid.pt", map_location="cpu"))
        self.hybrid.eval()

        bg_path = artifacts / "shap_background.npy"
        self.background = np.load(bg_path)
        self.explainer = shap.TreeExplainer(self.xgb, self.background)

    def _feature_vector(self, features: dict) -> np.ndarray:
        return np.array([[float(features[col]) for col in self.features]], dtype=np.float32)

    def predict_all(self, features: dict, sequence: list[int]) -> dict:
        x = self._feature_vector(features)

        rf_pred = float(self.rf.predict(x)[0])
        xgb_pred = float(self.xgb.predict(x)[0])

        x_tab = self.hybrid_scaler.transform(x)
        x_seq = np.array(sequence, dtype=np.float32)[None, None, :]
        with torch.no_grad():
            hybrid_pred = float(
                self.hybrid(
                    torch.tensor(x_tab, dtype=torch.float32),
                    torch.tensor(x_seq, dtype=torch.float32),
                ).numpy()[0]
            )

        ensemble = float(np.clip(0.2 * rf_pred + 0.3 * xgb_pred + 0.5 * hybrid_pred, 50, 99))

        shap_values = self.explainer.shap_values(x)
        contributions = []
        for idx, name in enumerate(self.features):
            impact = float(shap_values[0][idx])
            contributions.append(
                {
                    "feature_key": name,
                    "feature": FEATURE_LABELS.get(name, name.replace("_", " ").title()),
                    "impact": impact,
                    "direction": "improving sleep" if impact >= 0 else "hurting sleep",
                    "insight": build_human_insight(name, FEATURE_LABELS.get(name, name.replace("_", " ").title()), impact),
                    "value": float(x[0][idx]),
                }
            )
        contributions.sort(key=lambda it: abs(it["impact"]), reverse=True)

        return {
            "rf": rf_pred,
            "xgb": xgb_pred,
            "hybrid": hybrid_pred,
            "ensemble": ensemble,
            "top_factors": contributions[:8],
            "sleep_quality": "Good" if ensemble >= self.quality_threshold else "Poor",
        }


_model_bundle: ModelBundle | None = None


def get_model_bundle() -> ModelBundle:
    global _model_bundle
    if _model_bundle is None:
        _model_bundle = ModelBundle()
    return _model_bundle
