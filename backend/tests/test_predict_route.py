from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.session import Base, get_db
from app.db.models import PredictionRecord


@pytest.fixture
def test_database():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    yield engine, TestingSessionLocal

    app.dependency_overrides.clear()
    engine.dispose()


def make_payload():
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


def test_prediction_returns_results_and_persists_record(test_database):
    test_engine, TestingSessionLocal = test_database

    features = {"caffeine_mg_evening": 100}

    inactivity = SimpleNamespace(
        inferred_sleep_time="23:00"
    )

    top_factors = [
        {
            "feature_key": "caffeine_mg_evening",
            "feature": "Evening caffeine intake",
            "impact": -2.5,
            "direction": "hurting sleep",
            "insight": "Evening caffeine may affect sleep.",
            "value": 100,
        }
    ]

    fake_prediction = {
        "rf": 85.0,
        "xgb": 83.0,
        "hybrid": 84.0,
        "ensemble": 84.0,
        "sleep_quality": "Good",
        "top_factors": top_factors,
    }

    fake_bundle = MagicMock()
    fake_bundle.predict_all.return_value = fake_prediction

    recommendations = [
        "Consider reducing evening caffeine."
    ]

    with patch("app.main.engine", test_engine):
        with patch(
            "app.main.create_knowledge_service",
            return_value=MagicMock(),
        ):
            with patch(
                "app.api.routes.predict.build_engineered_features",
                return_value=(features, inactivity),
            ):
                with patch(
                    "app.api.routes.predict.get_model_bundle",
                    return_value=fake_bundle,
                ):
                   with patch(
                        "app.api.routes.predict.generate_recommendations",
                        return_value=recommendations,):
                        with patch(
                            # The route now runs the automatic recommendation
                            # pipeline, which also fills the explanation fields.
                            "app.api.routes.predict.recommend_for_prediction",
                            return_value={
                            "personalized_recommendations": None,
                            "generated_explanation": (
                                "Your estimated sleep efficiency is 84%. "
                                "Evening caffeine may affect your sleep."
                            ),
                            "explanation_status": "generated",
                        },
                     ) as mock_explanation:
                            with TestClient(app) as client:
                                response = client.post(
                                    "/api/v1/predict",
                                    json=make_payload(),
                                )

        mock_explanation.assert_called_once()

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_sleep_efficiency"] == 84.0
    assert result["sleep_quality_label"] == "Good"
    assert result["inferred_sleep_time"] == "23:00"
    assert result["model_outputs"]["ensemble"] == 84.0
    assert result["top_influencing_factors"] == top_factors
    assert result["recommendations"] == recommendations
    assert result["explanation_status"] == "generated"
    assert "84%" in result["generated_explanation"]
    assert "Evening caffeine" in result["generated_explanation"]

    with TestingSessionLocal() as db:
        record = db.query(PredictionRecord).one()

        assert record.predicted_efficiency == 84.0
        assert record.predicted_quality == "Good"
        assert record.top_factors == top_factors
        assert record.recommendations == recommendations


def test_prediction_uses_fallback_explanation_and_persists_record(test_database):
    test_engine, TestingSessionLocal = test_database

    features = {"caffeine_mg_evening": 100}

    inactivity = SimpleNamespace(
        inferred_sleep_time="23:00"
    )

    top_factors = [
        {
            "feature_key": "caffeine_mg_evening",
            "feature": "Evening caffeine intake",
            "impact": -2.5,
            "direction": "hurting sleep",
            "insight": "Evening caffeine may affect sleep.",
            "value": 100,
        }
    ]

    fake_prediction = {
        "rf": 85.0,
        "xgb": 83.0,
        "hybrid": 84.0,
        "ensemble": 84.0,
        "sleep_quality": "Good",
        "top_factors": top_factors,
    }

    fake_bundle = MagicMock()
    fake_bundle.predict_all.return_value = fake_prediction

    recommendations = [
        "Consider reducing evening caffeine."
    ]

    fallback_explanation = (
        "Your estimated sleep efficiency is 84%. "
        "Evening caffeine may affect your sleep. "
        "This estimate is based on lifestyle patterns and "
        "screen inactivity, which is an indirect indicator."
    )

    with patch("app.main.engine", test_engine):
        with patch(
            "app.main.create_knowledge_service",
            return_value=MagicMock(),
        ):
            with patch(
                "app.api.routes.predict.build_engineered_features",
                return_value=(features, inactivity),
            ):
                with patch(
                    "app.api.routes.predict.get_model_bundle",
                    return_value=fake_bundle,
                ):
                    with patch(
                        "app.api.routes.predict.generate_recommendations",
                        return_value=recommendations,
                    ):
                        with patch(
                            "app.api.routes.predict.recommend_for_prediction",
                            return_value={
                                "personalized_recommendations": None,
                                "generated_explanation": fallback_explanation,
                                "explanation_status": "fallback",
                            },
                        ):
                            with TestClient(app) as client:
                                response = client.post(
                                    "/api/v1/predict",
                                    json=make_payload(),
                                )

    assert response.status_code == 200

    result = response.json()

    # Original ML prediction must remain unchanged.
    assert result["predicted_sleep_efficiency"] == 84.0
    assert result["model_outputs"]["ensemble"] == 84.0
    assert result["sleep_quality_label"] == "Good"

    # Fallback explanation must be returned.
    assert result["explanation_status"] == "fallback"
    assert "84%" in result["generated_explanation"]
    assert "Evening caffeine" in result["generated_explanation"]

    # Prediction must still be persisted.
    with TestingSessionLocal() as db:
        record = db.query(PredictionRecord).one()

        assert record.predicted_efficiency == 84.0
        assert record.predicted_quality == "Good"
        assert record.top_factors == top_factors
        assert record.recommendations == recommendations




def test_prediction_rejects_invalid_lifestyle_input(test_database):
    test_engine, _ = test_database

    payload = make_payload()
    payload["lifestyle"]["stress_level"] = 15

    with patch("app.main.engine", test_engine):
        with patch(
            "app.main.create_knowledge_service",
            return_value=MagicMock(),
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/predict",
                    json=payload,
                )

    assert response.status_code == 422
