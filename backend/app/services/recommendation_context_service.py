"""Build a validated, model-grounded context for personalized recommendations.

This module turns the deployed models' own explanations into recommendation
signals. It never changes predictions or attributions. Key rules:

* TreeSHAP (RF, XGBoost) and Integrated Gradients (Hybrid) are reported side
  by side and never added together or averaged into an "ensemble SHAP".
* Only features in the leakage-screened ``safe9`` set from
  ``ml/experiments/hybrid_ablation_v2/report.md`` may drive a
  recommendation. Target-derived features (previous sleep, naps, mood) and
  sequence-derived features are reported but not used as drivers.
* A factor qualifies only when RF and XGBoost agree on the direction of its
  attribution and the attributions passed their reconstruction checks.
* Attribution describes model behaviour, not causation.
"""

from __future__ import annotations

import copy
from typing import Any

from app.services.model_service import FEATURE_LABELS

# Minimum mean |TreeSHAP| (percentage points) for a factor to be considered.
# A presentation threshold to skip negligible attributions, not a validated cut-off.
MIN_ATTRIBUTION = 0.1
MAX_RECOMMENDATIONS = 5

# Leakage-screened features (ablation v2 "safe9") with their knowledge-base
# topic. ``chunk_id`` is the knowledge entry that must be retrieved for the
# factor to receive scientific support; ``None`` means the knowledge base has
# no entry, so the factor cannot receive an evidence-backed recommendation.
RECOMMENDATION_TOPICS: dict[str, dict[str, Any]] = {
    "caffeine_mg_evening": {
        "chunk_id": "CAFF-001",
        # The generated action must mention one of these.
        "keywords": ("caffeine", "coffee", "tea", "energy drink", "decaf"),
        "query": "How does caffeine consumed in the evening affect sleep?",
        "unit": "mg",
        "title": "Move caffeine earlier in the day",
        "action": "Try having caffeine earlier in the day and avoiding it in the hours before bed.",
        "maintain": "Your current evening caffeine level is not lowering the prediction; keeping caffeine away from bedtime is reasonable.",
    },
    "alcohol_units": {
        "chunk_id": "ALC-001",
        # The generated action must mention one of these.
        "keywords": ("alcohol", "drink"),
        "query": "How does drinking alcohol before bed affect sleep quality?",
        "unit": "units",
        "title": "Limit alcohol close to bedtime",
        "action": "On evenings you drink, try having less and finishing earlier, well before going to bed.",
        "maintain": "Your current alcohol intake is not lowering the prediction; keeping it low near bedtime is reasonable.",
    },
    "dinner_hour": {
        "chunk_id": "MEAL-001",
        # The generated action must mention one of these.
        "keywords": ("dinner", "meal", "eat", "food"),
        "query": "Does eating a large meal close to bedtime affect sleep?",
        "unit": "hour of day",
        "title": "Leave more time between dinner and bed",
        "action": "Where practical, have your main evening meal earlier so it is not close to bedtime.",
        "maintain": "Your dinner timing is not lowering the prediction; keeping meals away from bedtime is reasonable.",
    },
    "stress_level": {
        "chunk_id": "STRESS-001",
        # The generated action must mention one of these.
        "keywords": ("stress", "relax", "calm", "wind", "breath", "mindful", "worr", "meditat"),
        "query": "How are stress and mood related to sleep?",
        "unit": "on a 1-10 scale",
        "title": "Build a short wind-down routine",
        "action": "Set aside a few minutes before bed for a calming routine such as slow breathing, journaling or light stretching.",
        "maintain": "Your reported stress level is not lowering the prediction; keeping a calm pre-sleep routine is reasonable.",
    },
    "physical_activity_min": {
        "chunk_id": "ACT-001",
        # The generated action must mention one of these.
        "keywords": ("activ", "exercise", "walk", "move", "movement", "sport"),
        "query": "How does regular physical activity affect sleep?",
        "unit": "minutes",
        "title": "Add regular daytime movement",
        "action": "Aim for regular moderate activity during the day, and finish vigorous exercise well before bedtime.",
        "maintain": "Your activity level is supporting the prediction; keeping up regular daytime activity is reasonable.",
    },
    "blue_light_minutes": {
        "chunk_id": "SCREEN-001",
        # The generated action must mention one of these.
        "keywords": ("screen", "light", "device", "phone", "tablet"),
        "query": "How does using screens and devices in the evening affect sleep?",
        "unit": "minutes",
        "title": "Reduce screen use before bed",
        "action": "Try putting screens away for the last part of the evening and using that time for screen-free activities.",
        "maintain": "Your evening screen time is not lowering the prediction; keeping screens away from bedtime is reasonable.",
    },
    "bedtime_consistency": {
        "chunk_id": "REG-001",
        # The generated action must mention one of these.
        "keywords": ("schedule", "bedtime", "regular", "consistent", "routine", "wake", "waking", "same time"),
        "query": "Why is a consistent sleep and wake schedule important?",
        "unit": "on a 0-1 consistency scale",
        "title": "Keep a more regular sleep schedule",
        "action": "Try to go to bed and wake up at similar times every day, including weekends.",
        "maintain": "Your bedtime consistency is supporting the prediction; keeping a regular schedule is reasonable.",
    },
    "work_hours": {"chunk_id": None},
    "hydration_liters": {"chunk_id": None},
}

TARGET_DERIVED = {
    "previous_sleep_hours": "Generated from the same sleep-duration field used to construct the target.",
    "naps_count": "Generated from the same sleep-duration field used to construct the target.",
    "naps_total_min": "Generated from the same sleep-duration field used to construct the target.",
    "nap_sleep_debt_proxy": "Computed from previous sleep hours and nap minutes, which are target-derived.",
    "mood_score": "Generated partly from the source sleep-quality field that also drives the target.",
}
SEQUENCE_DERIVED = {
    "inferred_sleep_start_hour",
    "longest_evening_inactivity_min",
    "nighttime_activity_bursts",
    "dinner_to_sleep_gap",
}
DUPLICATE_COMPOSITES = {"caffeine_stress_interaction", "activity_stress_balance"}


def exclusion_reason(feature_key: str) -> str | None:
    if feature_key in TARGET_DERIVED:
        return "Target-derived: " + TARGET_DERIVED[feature_key]
    if feature_key in SEQUENCE_DERIVED:
        return (
            "Sequence-derived: the training sequences were synthetic with an imposed sleep "
            "window, so this attribution is not a reliable basis for advice."
        )
    if feature_key in DUPLICATE_COMPOSITES:
        return "Deterministic combination of other inputs; its components are considered instead."
    if feature_key not in RECOMMENDATION_TOPICS:
        return "Not in the leakage-screened recommendation feature set."
    return None


def _validity(rf: dict, xgb: dict, hybrid: dict | None) -> dict[str, Any]:
    status = {
        "rf": {"method": "TreeSHAP", "valid": bool(rf["reconstruction"]["passed"]),
               "absolute_error": rf["reconstruction"]["absolute_error"]},
        "xgb": {"method": "TreeSHAP", "valid": bool(xgb["reconstruction"]["passed"]),
                "absolute_error": xgb["reconstruction"]["absolute_error"]},
    }
    if hybrid is not None:
        status["hybrid"] = {
            "method": "Integrated Gradients",
            "valid": bool(hybrid["completeness"]["passed"]),
            "absolute_error": hybrid["completeness"]["absolute_error"],
            "integration_steps": hybrid["completeness"]["integration_steps"],
        }
    return status


def _attribution_map(explanation: dict, key: str = "attributions") -> dict[str, float]:
    names = explanation["feature_names"]
    return dict(zip(names, explanation[key]))


def _top_rows(explanation_rows: list[dict], limit: int = 5) -> list[dict]:
    return [
        {
            "feature_key": row["feature_key"],
            "feature": row["feature"],
            "value": row["value"],
            "attribution": row["attribution"],
        }
        for row in explanation_rows[:limit]
    ]


def summarize_temporal(hybrid: dict | None, limit: int = 3) -> dict[str, Any] | None:
    """Aggregate the 288 five-minute IG attributions into hourly totals."""
    if hybrid is None:
        return None
    hourly: dict[int, dict[str, float]] = {}
    for position in hybrid["temporal"]["positions"]:
        hour = position["bin_index"] // 12
        bucket = hourly.setdefault(hour, {"attribution": 0.0, "active_bins": 0})
        bucket["attribution"] += position["attribution"]
        bucket["active_bins"] += int(position["activity_value"] > 0)
    rows = [
        {
            "hour": f"{hour:02d}:00-{(hour + 1) % 24:02d}:00",
            "attribution": values["attribution"],
            "active_five_minute_bins": values["active_bins"],
        }
        for hour, values in sorted(hourly.items())
    ]
    lowering = sorted((r for r in rows if r["attribution"] < 0), key=lambda r: r["attribution"])[:limit]
    raising = sorted((r for r in rows if r["attribution"] > 0), key=lambda r: -r["attribution"])[:limit]
    return {
        "method": "Integrated Gradients (Hybrid temporal branch), summed per hour",
        "hours_lowering_prediction": lowering,
        "hours_raising_prediction": raising,
        "note": (
            "Describes how the Hybrid model responded to this activity pattern. The model was "
            "trained on synthetic activity sequences, so these patterns are model behaviour, "
            "not evidence about real sleep. Inactivity is an indirect proxy, not proof of sleep."
        ),
    }


def build_recommendation_context(
    *,
    lifestyle: dict,
    features: dict,
    prediction: dict,
    inferred_sleep_time: str,
    explanations: dict[str, dict | None],
) -> dict[str, Any]:
    """Derive recommendation signals from validated model explanations.

    ``explanations`` holds the outputs of ``ModelBundle.explain_rf``,
    ``explain_xgb`` and ``explain_hybrid`` (Hybrid may be ``None``).
    """
    lifestyle = copy.deepcopy(lifestyle)
    rf, xgb, hybrid = explanations["rf"], explanations["xgb"], explanations.get("hybrid")
    validity = _validity(rf, xgb, hybrid)
    warnings: list[str] = list(prediction.get("diagnostics", {}).get("warnings", []))

    # Explanations must describe the same predictions the user received.
    for model_id in ("rf", "xgb"):
        if abs(explanations[model_id]["prediction"] - prediction[model_id]) > 1e-4:
            validity[model_id]["valid"] = False
            warnings.append(f"The {model_id} explanation did not match its prediction and was excluded.")
    if hybrid is not None and abs(hybrid["prediction"] - prediction["hybrid"]) > 1e-3:
        validity["hybrid"]["valid"] = False
        warnings.append("The Hybrid explanation did not match its prediction and was excluded.")

    tree_valid = validity["rf"]["valid"] and validity["xgb"]["valid"]
    hybrid_valid = hybrid is not None and validity["hybrid"]["valid"]
    if not tree_valid:
        warnings.append(
            "TreeSHAP explanations failed validation, so no model-grounded recommendations were produced."
        )

    rf_map, xgb_map = _attribution_map(rf), _attribution_map(xgb)
    hybrid_map = _attribution_map(hybrid["tabular"]) if hybrid_valid else {}

    factors, excluded, disagreements = [], [], []
    for feature_key in rf["feature_names"]:
        rf_value, xgb_value = rf_map[feature_key], xgb_map[feature_key]
        mean_abs = (abs(rf_value) + abs(xgb_value)) / 2
        reason = exclusion_reason(feature_key)
        if reason:
            if mean_abs >= MIN_ATTRIBUTION:
                excluded.append({"feature_key": feature_key, "feature": FEATURE_LABELS.get(feature_key, feature_key),
                                 "reason": reason})
            continue
        if mean_abs < MIN_ATTRIBUTION:
            continue
        agree = (rf_value > 0) == (xgb_value > 0) and rf_value != 0 and xgb_value != 0
        hybrid_value = hybrid_map.get(feature_key)
        entry = {
            "feature_key": feature_key,
            "feature": FEATURE_LABELS.get(feature_key, feature_key),
            "user_value": features.get(feature_key),
            "unit": RECOMMENDATION_TOPICS[feature_key].get("unit", ""),
            "rf_treeshap": rf_value,
            "xgb_treeshap": xgb_value,
            "hybrid_integrated_gradients": hybrid_value,
            "mean_abs_treeshap": mean_abs,
            "direction": "lowers" if rf_value < 0 else "raises",
            "rf_xgb_agree": agree,
            "hybrid_agrees": None if hybrid_value is None else (hybrid_value < 0) == (rf_value < 0),
            "has_knowledge_topic": RECOMMENDATION_TOPICS[feature_key]["chunk_id"] is not None,
        }
        if agree:
            factors.append(entry)
        else:
            disagreements.append(entry)

    factors.sort(key=lambda item: item["mean_abs_treeshap"], reverse=True)
    lowering = [f for f in factors if f["direction"] == "lowers"]
    raising = [f for f in factors if f["direction"] == "raises"]

    candidates: list[dict] = []
    if tree_valid:
        actionable_lowering = [f for f in lowering if f["has_knowledge_topic"]]
        actionable_raising = [f for f in raising if f["has_knowledge_topic"]]
        candidates = actionable_lowering[:MAX_RECOMMENDATIONS]
        if len(candidates) < 3:
            candidates += actionable_raising[: 3 - len(candidates)]

    disagreement = prediction.get("diagnostics", {}).get("model_disagreement", {})
    uncertainty = []
    if disagreement.get("level") in {"moderate", "high"}:
        uncertainty.append(
            f"The three models disagree by {disagreement['spread']:.2f} percentage points, "
            "so the estimate is uncertain."
        )
    if disagreements:
        uncertainty.append(
            "RF and XGBoost disagree on the direction of: "
            + ", ".join(item["feature"] for item in disagreements)
            + "; these were not used for recommendations."
        )
    without_topic = [f["feature"] for f in lowering if not f["has_knowledge_topic"]]
    if without_topic:
        uncertainty.append(
            "No knowledge-base topic exists for " + ", ".join(without_topic)
            + ", so no evidence-backed recommendation is given for it."
        )
    uncertainty.append(
        "Feature attributions describe how the models behave, not what causes sleep changes; "
        "changing a factor is not guaranteed to change real sleep efficiency."
    )

    return {
        "prediction_context": {
            "lifestyle_inputs": lifestyle,
            "engineered_features": {key: float(value) for key, value in features.items()},
            "model_predictions": {key: float(prediction[key]) for key in ("rf", "xgb", "hybrid", "ensemble")},
            "predicted_sleep_efficiency": float(prediction["ensemble"]),
            "sleep_quality_label": prediction["sleep_quality"],
            "inferred_sleep_time": inferred_sleep_time,
            "model_disagreement": disagreement,
        },
        "model_explanation": {
            "validity": validity,
            "rf_treeshap": {
                "baseline": rf["baseline"]["value"],
                "top_negative": _top_rows(rf["negative_contributions"]),
                "top_positive": _top_rows(rf["positive_contributions"]),
            },
            "xgb_treeshap": {
                "baseline": xgb["baseline"]["value"],
                "top_negative": _top_rows(xgb["negative_contributions"]),
                "top_positive": _top_rows(xgb["positive_contributions"]),
            },
            "hybrid_integrated_gradients": None if not hybrid_valid else {
                "baseline_prediction": hybrid["baseline"]["prediction"],
                "baseline_description": {
                    "tabular": hybrid["baseline"]["tabular"],
                    "temporal": hybrid["baseline"]["temporal"],
                },
                "top_negative_tabular": _top_rows(hybrid["tabular"]["negative_contributions"]),
                "top_positive_tabular": _top_rows(hybrid["tabular"]["positive_contributions"]),
                "temporal_summary": summarize_temporal(hybrid),
            },
            "agreed_lowering_factors": lowering,
            "agreed_raising_factors": raising,
            "rf_xgb_disagreements": disagreements,
            "excluded_features": excluded,
            "method_note": (
                "TreeSHAP (RF, XGBoost) and Integrated Gradients (Hybrid) are reported separately "
                "and are not combined. No ensemble attribution exists."
            ),
        },
        "candidates": candidates,
        "uncertainty": uncertainty,
        "warnings": warnings,
    }
