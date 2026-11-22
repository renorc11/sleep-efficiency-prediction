from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

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


# Deployed ensemble policy. Retained unchanged for backward compatibility; see
# ml/reports/ensemble_development_selection_v1.md for evidence that it
# underperforms RF in development comparisons.
ENSEMBLE_WEIGHTS = {"rf": 0.2, "xgb": 0.3, "hybrid": 0.5}
ENSEMBLE_CLIP_RANGE = (50.0, 99.0)

# Output-domain policy. Sleep efficiency is a percentage, so any value outside
# 0-100 is not a valid efficiency. Individual model outputs are never altered;
# out-of-domain values are reported with a warning instead.
VALID_EFFICIENCY_RANGE = (0.0, 100.0)

# Descriptive spread bands for model disagreement, in percentage points. These
# are presentation thresholds, not validated uncertainty intervals. For scale,
# the historical RF test MAE was 2.64 and the Hybrid's was 4.98.
DISAGREEMENT_BANDS = ((5.0, "low"), (10.0, "moderate"))


def build_prediction_diagnostics(
    rf: float, xgb: float, hybrid: float, ensemble_unclipped: float, ensemble: float
) -> dict[str, Any]:
    """Summarize output-domain checks, ensemble clipping, and model disagreement."""
    low, high = VALID_EFFICIENCY_RANGE
    raw = {"rf": rf, "xgb": xgb, "hybrid": hybrid, "ensemble_unclipped": ensemble_unclipped}
    out_of_domain = [name for name, value in raw.items() if not low <= value <= high]
    warnings = [
        f"The {name} output {raw[name]:.2f} is outside the valid {low:.0f}-{high:.0f}% "
        "sleep-efficiency domain and is shown unmodified for diagnostics only."
        for name in out_of_domain
    ]

    clipped = ensemble != ensemble_unclipped
    if clipped:
        warnings.append(
            f"The deployed ensemble value {ensemble_unclipped:.2f} was clipped to "
            f"{ensemble:.2f} by the fixed {ENSEMBLE_CLIP_RANGE[0]:.0f}-"
            f"{ENSEMBLE_CLIP_RANGE[1]:.0f} policy."
        )

    individual = [rf, xgb, hybrid]
    spread = max(individual) - min(individual)
    level = next((label for limit, label in DISAGREEMENT_BANDS if spread <= limit), "high")
    if level != "low":
        warnings.append(
            f"The three models disagree by {spread:.2f} percentage points ({level}); "
            "treat the final estimate as uncertain."
        )

    return {
        "raw_outputs": raw,
        "valid_domain": {"min": low, "max": high, "out_of_domain_models": out_of_domain},
        "ensemble_policy": {
            "weights": dict(ENSEMBLE_WEIGHTS),
            "clip_range": list(ENSEMBLE_CLIP_RANGE),
            "clipping_applied": clipped,
            "note": "Fixed deployed policy retained for backward compatibility; it is not "
            "the best-validated option in development comparisons.",
        },
        "model_disagreement": {
            "spread": spread,
            "rf_xgb_abs_difference": abs(rf - xgb),
            "level": level,
            "bands": {label: limit for limit, label in DISAGREEMENT_BANDS},
            "note": "Descriptive presentation bands, not validated uncertainty intervals.",
        },
        "warnings": warnings,
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
        # Keep the original XGBoost explainer alias for the deployed prediction
        # response, while providing model-specific explainers for diagnostics.
        self.xgb_explainer = shap.TreeExplainer(self.xgb, self.background)
        self.rf_explainer: shap.TreeExplainer | None = None
        self._rf_explainer_lock = threading.Lock()
        self.explainer = self.xgb_explainer
        self._global_explanations: dict[str, dict[str, Any]] = {}
        self._global_explanations_lock = threading.Lock()

    def _get_rf_explainer(self) -> shap.TreeExplainer:
        """Construct RF SHAP lazily so ordinary prediction initialization is unchanged."""
        if self.rf_explainer is None:
            with self._rf_explainer_lock:
                if self.rf_explainer is None:
                    self.rf_explainer = shap.TreeExplainer(self.rf, self.background)
        return self.rf_explainer

    def _feature_vector(self, features: dict) -> np.ndarray:
        missing = [col for col in self.features if col not in features]
        if missing:
            raise ValueError(f"Missing required engineered features: {', '.join(missing)}")
        vector = np.array([[float(features[col]) for col in self.features]], dtype=np.float32)
        if not np.isfinite(vector).all():
            bad = [col for col, value in zip(self.features, vector[0]) if not np.isfinite(value)]
            raise ValueError(f"Engineered features must be finite: {', '.join(bad)}")
        return vector

    def _validate_sequence(self, sequence: list[int] | np.ndarray) -> np.ndarray:
        array = np.asarray(sequence, dtype=np.float32)
        if array.ndim == 1:
            if array.shape[0] != 288:
                raise ValueError("The activity sequence must contain exactly 288 five-minute bins.")
            array = array[None, None, :]
        elif not (array.ndim == 3 and array.shape == (1, 1, 288)):
            raise ValueError("Activity sequence must have shape [288] or [1, 1, 288].")
        if not np.isin(array, (0.0, 1.0)).all():
            raise ValueError("Activity sequence values must be binary (0 or 1).")
        return array

    @staticmethod
    def _ensemble(rf: float, xgb: float, hybrid: float) -> tuple[float, float]:
        """Return the deployed ensemble before and after its fixed final clipping."""
        unclipped = float(
            ENSEMBLE_WEIGHTS["rf"] * rf
            + ENSEMBLE_WEIGHTS["xgb"] * xgb
            + ENSEMBLE_WEIGHTS["hybrid"] * hybrid
        )
        return unclipped, float(np.clip(unclipped, *ENSEMBLE_CLIP_RANGE))

    @staticmethod
    def _attribution_lists(items: list[dict[str, Any]], key: str = "attribution") -> dict[str, list[dict[str, Any]]]:
        ranked = sorted(items, key=lambda item: abs(float(item[key])), reverse=True)
        return {
            "positive_contributions": [item for item in ranked if float(item[key]) > 0],
            "negative_contributions": [item for item in ranked if float(item[key]) < 0],
            "ranked_contributions": ranked,
        }

    def _tree_shap_local(
        self,
        *,
        model_id: str,
        explainer: shap.TreeExplainer,
        x: np.ndarray,
        prediction: float,
    ) -> dict[str, Any]:
        values = np.asarray(explainer.shap_values(x), dtype=float)
        if values.ndim == 3:
            values = values[..., 0]
        if values.shape != (1, len(self.features)):
            raise RuntimeError(f"Unexpected {model_id} SHAP shape: {values.shape}")

        base_value = float(np.asarray(explainer.expected_value, dtype=float).reshape(-1)[0])
        rows = [
            {
                "feature_key": feature,
                "feature": FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
                "value": float(x[0, index]),
                "attribution": float(values[0, index]),
            }
            for index, feature in enumerate(self.features)
        ]
        reconstruction = base_value + float(values.sum())
        error = float(prediction - reconstruction)
        return {
            "model": model_id,
            "method": "TreeSHAP",
            "prediction": prediction,
            "baseline": {"value": base_value, "description": "TreeSHAP expected model output."},
            "feature_names": self.features,
            "feature_values": [float(value) for value in x[0]],
            "attributions": [float(value) for value in values[0]],
            "reconstruction": {
                "reconstructed_prediction": reconstruction,
                "absolute_error": abs(error),
                "tolerance": 0.001,
                "passed": abs(error) <= 0.001,
            },
            **self._attribution_lists(rows),
            "warnings": [],
        }

    def explain_rf(self, features: dict) -> dict[str, Any]:
        x = self._feature_vector(features)
        prediction = float(self.rf.predict(x)[0])
        return self._tree_shap_local(
            model_id="rf", explainer=self._get_rf_explainer(), x=x, prediction=prediction
        )

    def explain_xgb(self, features: dict) -> dict[str, Any]:
        x = self._feature_vector(features)
        prediction = float(self.xgb.predict(x)[0])
        return self._tree_shap_local(
            model_id="xgb", explainer=self.xgb_explainer, x=x, prediction=prediction
        )

    @staticmethod
    def _bin_interval(index: int) -> str:
        start_minutes = index * 5
        end_minutes = start_minutes + 5
        return (
            f"{start_minutes // 60:02d}:{start_minutes % 60:02d}-"
            f"{end_minutes // 60:02d}:{end_minutes % 60:02d}"
        )

    def explain_hybrid(
        self,
        features: dict,
        sequence: list[int] | np.ndarray,
        *,
        steps: int = 64,
    ) -> dict[str, Any]:
        """Explain the deployed hybrid with direct, two-input Integrated Gradients."""
        if steps < 2:
            raise ValueError("Integrated Gradients requires at least two integration steps.")

        x = self._feature_vector(features)
        x_tab = self.hybrid_scaler.transform(x).astype(np.float32)
        x_seq = self._validate_sequence(sequence)

        actual_tab = torch.tensor(x_tab, dtype=torch.float32)
        actual_seq = torch.tensor(x_seq, dtype=torch.float32)
        baseline_tab = torch.zeros_like(actual_tab)
        baseline_seq = torch.zeros_like(actual_seq)

        with torch.no_grad():
            prediction = float(self.hybrid(actual_tab, actual_seq).item())
            baseline_prediction = float(self.hybrid(baseline_tab, baseline_seq).item())

        alphas = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float32).view(-1, 1)
        tab_path = (baseline_tab + alphas * (actual_tab - baseline_tab)).detach().requires_grad_(True)
        seq_path = (
            baseline_seq
            + alphas.view(-1, 1, 1) * (actual_seq - baseline_seq)
        ).detach().requires_grad_(True)

        outputs = self.hybrid(tab_path, seq_path)
        gradients_tab, gradients_seq = torch.autograd.grad(
            outputs=outputs.sum(), inputs=(tab_path, seq_path)
        )
        weights = torch.ones(steps + 1, dtype=torch.float32)
        weights[0] = weights[-1] = 0.5
        average_tab_gradient = (gradients_tab * weights.view(-1, 1)).sum(dim=0) / steps
        average_seq_gradient = (gradients_seq * weights.view(-1, 1, 1)).sum(dim=0) / steps
        tabular_attributions = ((actual_tab - baseline_tab) * average_tab_gradient)[0].detach().numpy()
        temporal_attributions = ((actual_seq - baseline_seq) * average_seq_gradient)[0, 0].detach().numpy()

        tabular_rows = [
            {
                "feature_key": feature,
                "feature": FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
                "value": float(x[0, index]),
                "attribution": float(tabular_attributions[index]),
            }
            for index, feature in enumerate(self.features)
        ]
        temporal_rows = [
            {
                "bin_index": index,
                "time_interval": self._bin_interval(index),
                "activity_value": float(x_seq[0, 0, index]),
                "attribution": float(temporal_attributions[index]),
            }
            for index in range(288)
        ]
        reconstructed = baseline_prediction + float(tabular_attributions.sum() + temporal_attributions.sum())
        error = float(prediction - reconstructed)

        return {
            "model": "hybrid",
            "method": "IntegratedGradients",
            "prediction": prediction,
            "baseline": {
                "prediction": baseline_prediction,
                "tabular": "All-zero standardized tabular vector, representing the scaler training mean.",
                "temporal": "All-zero synthetic activity sequence, representing no interaction in every five-minute bin.",
            },
            "tabular": {
                "feature_names": self.features,
                "feature_values": [float(value) for value in x[0]],
                # Integrated Gradients is computed in the scaler's standardized
                # space; raw values above are for display, these are what the
                # network actually received.
                "standardized_values": [float(value) for value in x_tab[0]],
                "value_note": "feature_values are raw engineered inputs; attributions were computed on standardized_values (StandardScaler fitted on training rows).",
                "attributions": [float(value) for value in tabular_attributions],
                **self._attribution_lists(tabular_rows),
            },
            "temporal": {
                "bin_minutes": 5,
                "positions": temporal_rows,
                **self._attribution_lists(temporal_rows),
            },
            "completeness": {
                "reconstructed_prediction": reconstructed,
                "absolute_error": abs(error),
                "tolerance": 0.5,
                "passed": abs(error) <= 0.5,
                "integration_steps": steps,
                "description": "Completeness applies to combined tabular and temporal attributions, not either branch alone.",
            },
            "warnings": [
                "Integrated Gradients attributes the deployed model; it is not a SHAP value.",
                "Temporal inputs are synthetic activity/inactivity sequences, not observed device or wearable measurements.",
            ]
            + (
                [
                    f"The Hybrid output {prediction:.2f} is outside the valid "
                    f"{VALID_EFFICIENCY_RANGE[0]:.0f}-{VALID_EFFICIENCY_RANGE[1]:.0f}% domain; "
                    "its attributions explain an invalid percentage."
                ]
                if not VALID_EFFICIENCY_RANGE[0] <= prediction <= VALID_EFFICIENCY_RANGE[1]
                else []
            ),
        }

    def global_tree_shap(self, model_id: str, sample_size: int = 100) -> dict[str, Any]:
        if model_id not in {"rf", "xgb"}:
            raise ValueError("Global TreeSHAP is available only for rf and xgb.")
        if sample_size < 1:
            raise ValueError("sample_size must be positive.")
        cache_key = f"{model_id}:{sample_size}"
        with self._global_explanations_lock:
            if cache_key in self._global_explanations:
                return self._global_explanations[cache_key]

            data_path = settings.artifacts_dir.parent / "data" / "processed" / "dataset_engineered.csv"
            split_path = settings.artifacts_dir.parent / "data" / "processed" / "split.json"
            import pandas as pd

            frame = pd.read_csv(data_path)
            with split_path.open("r", encoding="utf-8") as handle:
                split = json.load(handle)
            # Split indices are row positions (np.arange in preprocess.py), so
            # select with .iloc rather than label-based .loc.
            development_indices = [int(i) for i in split["train_idx"]] + [int(i) for i in split["val_idx"]]
            if not development_indices:
                raise RuntimeError("No development indices are available for global explainability.")
            test_indices = {int(i) for i in split["test_idx"]}
            if test_indices.intersection(development_indices):
                raise RuntimeError("Split metadata overlaps test rows with development rows.")
            if max(development_indices) >= len(frame):
                raise RuntimeError("Split metadata references rows outside the engineered dataset.")
            selected = np.linspace(
                0, len(development_indices) - 1, num=min(sample_size, len(development_indices)), dtype=int
            )
            positions = [development_indices[index] for index in selected]
            x = frame.iloc[positions][self.features].to_numpy(dtype=np.float32)
            explainer = self._get_rf_explainer() if model_id == "rf" else self.xgb_explainer
            values = np.asarray(explainer.shap_values(x), dtype=float)
            if values.ndim == 3:
                values = values[..., 0]
            if values.shape != x.shape:
                raise RuntimeError(f"Unexpected global {model_id} SHAP shape: {values.shape}")
            mean_abs = np.abs(values).mean(axis=0)
            ranked = sorted(
                (
                    {
                        "feature_key": feature,
                        "feature": FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
                        "mean_absolute_attribution": float(mean_abs[index]),
                        # Retained for clients of the original /explain/global route.
                        "importance": float(mean_abs[index]),
                    }
                    for index, feature in enumerate(self.features)
                ),
                key=lambda row: row["mean_absolute_attribution"],
                reverse=True,
            )
            for rank, row in enumerate(ranked, start=1):
                row["rank"] = rank
            result = {
                "model": model_id,
                "method": "TreeSHAP mean absolute attribution",
                "sample_count": int(x.shape[0]),
                "sample_source": "Deterministic evenly spaced sample from existing train plus validation rows.",
                "sample_row_positions": positions,
                "excluded_test_row_count": len(test_indices),
                "feature_importance": ranked,
                "warnings": [
                    "This is a development-data descriptive explanation, not a new model evaluation.",
                ],
            }
            self._global_explanations[cache_key] = result
            return result

    def explain_all(self, features: dict, sequence: list[int], *, ig_steps: int = 64) -> dict[str, Any]:
        x = self._feature_vector(features)
        x_seq = self._validate_sequence(sequence)
        with torch.no_grad():
            hybrid_prediction = float(
                self.hybrid(
                    torch.tensor(self.hybrid_scaler.transform(x), dtype=torch.float32),
                    torch.tensor(x_seq, dtype=torch.float32),
                ).item()
            )
        rf_prediction = float(self.rf.predict(x)[0])
        xgb_prediction = float(self.xgb.predict(x)[0])
        ensemble_unclipped, ensemble = self._ensemble(rf_prediction, xgb_prediction, hybrid_prediction)
        diagnostics = build_prediction_diagnostics(
            rf_prediction, xgb_prediction, hybrid_prediction, ensemble_unclipped, ensemble
        )
        return {
            "model_predictions": {
                "rf": rf_prediction,
                "xgb": xgb_prediction,
                "hybrid": hybrid_prediction,
                "ensemble": ensemble,
            },
            "prediction_diagnostics": diagnostics,
            "model_explanations": {
                "rf": self.explain_rf(features),
                "xgb": self.explain_xgb(features),
                "hybrid": self.explain_hybrid(features, x_seq, steps=ig_steps),
            },
            "warnings": [
                "The ensemble output is retained for backward compatibility and is not given a combined attribution because TreeSHAP and Integrated Gradients are not directly interchangeable.",
                "The 288-bin temporal input is synthetic-development data and must not be interpreted as real device or wearable activity.",
            ],
        }

    def predict_all(self, features: dict, sequence: list[int]) -> dict:
        x = self._feature_vector(features)

        rf_pred = float(self.rf.predict(x)[0])
        xgb_pred = float(self.xgb.predict(x)[0])

        x_tab = self.hybrid_scaler.transform(x)
        x_seq = self._validate_sequence(sequence)
        with torch.no_grad():
            hybrid_pred = float(
                self.hybrid(
                    torch.tensor(x_tab, dtype=torch.float32),
                    torch.tensor(x_seq, dtype=torch.float32),
                ).numpy()[0]
            )

        ensemble_unclipped, ensemble = self._ensemble(rf_pred, xgb_pred, hybrid_pred)

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
            "diagnostics": build_prediction_diagnostics(
                rf_pred, xgb_pred, hybrid_pred, ensemble_unclipped, ensemble
            ),
        }


_model_bundle: ModelBundle | None = None
_model_bundle_lock = threading.Lock()


def get_model_bundle() -> ModelBundle:
    """Load the artifacts once, even when the first requests arrive concurrently."""
    global _model_bundle
    if _model_bundle is None:
        with _model_bundle_lock:
            if _model_bundle is None:
                _model_bundle = ModelBundle()
    return _model_bundle
