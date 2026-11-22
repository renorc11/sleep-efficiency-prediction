from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.api.routes.explain import local_explanation
from app.schemas.prediction import PredictionRequest


def _payload():
    return {
        "lifestyle": {
            "caffeine_mg_evening": 100,
            "alcohol_units": 0,
            "dinner_hour": 20,
            "stress_level": 5,
            "mood_score": 1,
            "physical_activity_min": 30,
            "naps_count": 1,
            "naps_total_min": 20,
            "previous_sleep_hours": 7,
            "work_hours": 8,
            "blue_light_minutes": 60,
            "bedtime_consistency": 0.8,
            "hydration_liters": 2,
        },
        "screen_activity_sequence": [0] * 288,
        "sleep_time_hint": "23:00",
    }


def test_local_explanation_returns_unified_model_contract():
    bundle = MagicMock()
    bundle.explain_all.return_value = {
        "model_predictions": {"rf": 81.0, "xgb": 82.0, "hybrid": 80.0, "ensemble": 81.0},
        "model_explanations": {"rf": {"method": "TreeSHAP"}, "xgb": {"method": "TreeSHAP"}, "hybrid": {"method": "IntegratedGradients"}},
        "warnings": ["Synthetic temporal input."],
    }

    with patch(
        "app.api.routes.explain.build_engineered_features",
        return_value=({"caffeine_mg_evening": 100}, SimpleNamespace(inferred_sleep_time="23:00")),
    ):
        with patch("app.api.routes.explain.get_model_bundle", return_value=bundle):
            response = local_explanation(PredictionRequest.model_validate(_payload()))

    assert response.inferred_sleep_time == "23:00"
    assert response.model_explanations["hybrid"]["method"] == "IntegratedGradients"
    bundle.explain_all.assert_called_once()
