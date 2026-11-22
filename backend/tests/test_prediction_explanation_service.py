
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.services.prediction_explanation_service import (
    build_retrieval_query,
    build_fallback_explanation,
    generate_prediction_explanation,
    validate_generated_explanation,
)


@pytest.fixture
def prediction():
    return {
        "rf": 85.0,
        "xgb": 83.0,
        "hybrid": 84.0,
        "ensemble": 84.0,
        "sleep_quality": "Good",
    }


@pytest.fixture
def lifestyle():
    return {
        "caffeine_mg_evening": 100,
        "alcohol_units": 0,
        "dinner_hour": 20,
        "stress_level": 5,
        "mood_score": 1,
        "physical_activity_min": 30,
    }


@pytest.fixture
def top_factors():
    return [
        {
            "feature_key": "caffeine_mg_evening",
            "feature": "Evening caffeine intake",
            "impact": -2.5,
            "direction": "hurting sleep",
            "insight": "Evening caffeine may affect sleep.",
            "value": 100,
        }
    ]


@pytest.fixture
def recommendations():
    return [
        "Consider reducing evening caffeine.",
    ]


@pytest.fixture
def knowledge_service():
    service = MagicMock()
    service.retrieve.return_value = {
        "dataset_version": "0.3.1",
        "chunks": [
            {
                "chunk_id": "CAFF-001",
                "topic": "Caffeine and sleep",
                "content": (
                    "Evening caffeine consumption "
                    "may interfere with sleep."
                ),
                "similarity": 0.90,
                "source_ids": ["SRC-001"],
            }
        ],
        "sources": [
            {
                "source_id": "SRC-001",
                "title": "Caffeine and Sleep",
            }
        ],
    }
    return service


def make_model_response(content):
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                }
            }
        ]
    }


def test_build_retrieval_query(lifestyle, top_factors):
    query = build_retrieval_query(
        lifestyle=lifestyle,
        top_factors=top_factors,
    )

    assert "Evening caffeine intake" in query
    assert "Evening caffeine may affect sleep." in query
    assert "caffeine_mg_evening" in query


def test_fallback_explanation_contains_prediction_and_factors(
    prediction,
    top_factors,
    recommendations,
):
    explanation = build_fallback_explanation(
        prediction=prediction,
        top_factors=top_factors,
        recommendations=recommendations,
        inferred_sleep_time="23:00",
    )

    assert "84.00%" in explanation
    assert "23:00" in explanation
    assert "Evening caffeine may affect sleep." in explanation
    assert "Consider reducing evening caffeine." in explanation
    assert "not a direct measurement" in explanation


@patch("app.services.prediction_explanation_service.httpx.Client")
def test_prediction_explanation_generated_successfully(
    mock_client_class,
    prediction,
    lifestyle,
    top_factors,
    recommendations,
    knowledge_service,
):
    mock_client = MagicMock()
    mock_client_class.return_value.__enter__.return_value = mock_client

    mock_response = MagicMock()
    mock_response.json.return_value = make_model_response(
        "Your estimated sleep efficiency is 84%. "
        "Evening caffeine may be an important factor."
    )
    mock_client.post.return_value = mock_response

    original_prediction = prediction.copy()

    result = generate_prediction_explanation(
        prediction=prediction,
        lifestyle=lifestyle,
        top_factors=top_factors,
        recommendations=recommendations,
        inferred_sleep_time="23:00",
        knowledge_service=knowledge_service,
    )

    assert result["explanation_status"] == "generated"
    assert "84%" in result["generated_explanation"]

    knowledge_service.retrieve.assert_called_once()

    assert knowledge_service.retrieve.call_args.kwargs == {
        "top_k": 3
    }

    mock_client.post.assert_called_once()

    # The original ML prediction must remain unchanged.
    assert prediction == original_prediction


def test_no_relevant_evidence_returns_fallback(
    prediction,
    lifestyle,
    top_factors,
    recommendations,
    knowledge_service,
):
    knowledge_service.retrieve.return_value = {
        "dataset_version": "0.3.1",
        "chunks": [],
        "sources": [],
    }

    with patch(
        "app.services.prediction_explanation_service.httpx.Client"
    ) as mock_client_class:

        result = generate_prediction_explanation(
            prediction=prediction,
            lifestyle=lifestyle,
            top_factors=top_factors,
            recommendations=recommendations,
            inferred_sleep_time="23:00",
            knowledge_service=knowledge_service,
        )

    assert result["explanation_status"] == "fallback"
    assert "84.00%" in result["generated_explanation"]
    mock_client_class.assert_not_called()


def test_retrieval_failure_returns_fallback(
    prediction,
    lifestyle,
    top_factors,
    recommendations,
    knowledge_service,
):
    knowledge_service.retrieve.side_effect = RuntimeError(
        "Knowledge retrieval failed."
    )

    result = generate_prediction_explanation(
        prediction=prediction,
        lifestyle=lifestyle,
        top_factors=top_factors,
        recommendations=recommendations,
        inferred_sleep_time="23:00",
        knowledge_service=knowledge_service,
    )

    assert result["explanation_status"] == "fallback"
    assert "84.00%" in result["generated_explanation"]


@patch("app.services.prediction_explanation_service.httpx.Client")
def test_biomistral_timeout_returns_fallback(
    mock_client_class,
    prediction,
    lifestyle,
    top_factors,
    recommendations,
    knowledge_service,
):
    mock_client = MagicMock()
    mock_client_class.return_value.__enter__.return_value = mock_client

    mock_client.post.side_effect = httpx.TimeoutException(
        "Request timed out."
    )

    result = generate_prediction_explanation(
        prediction=prediction,
        lifestyle=lifestyle,
        top_factors=top_factors,
        recommendations=recommendations,
        inferred_sleep_time="23:00",
        knowledge_service=knowledge_service,
    )

    assert result["explanation_status"] == "fallback"
    assert "84.00%" in result["generated_explanation"]


@patch("app.services.prediction_explanation_service.httpx.Client")
def test_empty_biomistral_response_returns_fallback(
    mock_client_class,
    prediction,
    lifestyle,
    top_factors,
    recommendations,
    knowledge_service,
):
    mock_client = MagicMock()
    mock_client_class.return_value.__enter__.return_value = mock_client

    mock_response = MagicMock()
    mock_response.json.return_value = make_model_response("   ")
    mock_client.post.return_value = mock_response

    result = generate_prediction_explanation(
        prediction=prediction,
        lifestyle=lifestyle,
        top_factors=top_factors,
        recommendations=recommendations,
        inferred_sleep_time="23:00",
        knowledge_service=knowledge_service,
    )

    assert result["explanation_status"] == "fallback"
    assert "84.00%" in result["generated_explanation"]


@patch("app.services.prediction_explanation_service.httpx.Client")
def test_http_error_returns_fallback(
    mock_client_class,
    prediction,
    lifestyle,
    top_factors,
    recommendations,
    knowledge_service,
):
    mock_client = MagicMock()
    mock_client_class.return_value.__enter__.return_value = mock_client

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = (
        httpx.HTTPStatusError(
            "Server error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )
    )
    mock_client.post.return_value = mock_response

    result = generate_prediction_explanation(
        prediction=prediction,
        lifestyle=lifestyle,
        top_factors=top_factors,
        recommendations=recommendations,
        inferred_sleep_time="23:00",
        knowledge_service=knowledge_service,
    )

    assert result["explanation_status"] == "fallback"
    assert "84.00%" in result["generated_explanation"]

@pytest.fixture
def validation_data():
    prediction = {
        "ensemble": 58.99,
    }

    top_factors = [
        {"feature": "Stress level"},
        {"feature": "Previous night's sleep duration"},
        {"feature": "Physical activity"},
    ]

    return prediction, top_factors


def test_valid_explanation_passes_validation(validation_data):
    prediction, top_factors = validation_data

    content = (
        "Your estimated sleep efficiency is 58.99%. "
        "Stress level, previous night's sleep, and "
        "physical activity are important factors."
    )

    is_valid, reason = validate_generated_explanation(
        content, prediction, top_factors
    )

    assert is_valid is True


def test_incorrect_percentage_is_rejected(validation_data):
    prediction, top_factors = validation_data

    content = (
        "Your sleep efficiency is 84%. "
        "Stress level, previous night's sleep, "
        "and physical activity are important."
    )

    is_valid, reason = validate_generated_explanation(
        content, prediction, top_factors
    )

    assert is_valid is False
    assert reason == "Prediction value is incorrect."


def test_missing_percentage_is_rejected(validation_data):
    prediction, top_factors = validation_data

    content = (
        "Stress level, previous night's sleep, "
        "and physical activity are important."
    )

    is_valid, reason = validate_generated_explanation(
        content, prediction, top_factors
    )

    assert is_valid is False
    assert reason == "Prediction percentage is missing."


def test_unresolved_placeholder_is_rejected(validation_data):
    prediction, top_factors = validation_data

    content = (
        "Your sleep efficiency is 58.99%. "
        "Hello [user_email]. Stress level, previous night's "
        "sleep, and physical activity are important."
    )

    is_valid, reason = validate_generated_explanation(
        content, prediction, top_factors
    )

    assert is_valid is False
    assert reason == "Unresolved placeholder found."


def test_email_format_is_rejected(validation_data):
    prediction, top_factors = validation_data

    content = (
        "Subject: Your Sleep Report\n"
        "Your sleep efficiency is 58.99%. "
        "Stress level, previous night's sleep, "
        "and physical activity are important."
    )

    is_valid, reason = validate_generated_explanation(
        content, prediction, top_factors
    )

    assert is_valid is False
    assert reason == "Unexpected email format."


def test_missing_shap_factor_is_rejected(validation_data):
    prediction, top_factors = validation_data

    content = (
        "Your sleep efficiency is 58.99%. "
        "Stress level and previous night's sleep "
        "are important."
    )

    is_valid, reason = validate_generated_explanation(
        content, prediction, top_factors
    )

    assert is_valid is False
    assert "Physical activity" in reason


@pytest.mark.parametrize(
    "percentage",
    ["58.99%", "58.990%", "58.9900%"],
)
def test_valid_percentage_formats_are_accepted(
    validation_data, percentage
):
    prediction, top_factors = validation_data

    content = (
        f"Your sleep efficiency is {percentage}. "
        "Stress level, previous night's sleep, "
        "and physical activity are important."
    )

    is_valid, reason = validate_generated_explanation(
        content, prediction, top_factors
    )

    assert is_valid is True

def test_rounded_incorrect_percentage_is_rejected(validation_data):
    prediction, top_factors = validation_data

    content = (
        "Your sleep efficiency is 58.9%. "
        "Stress level, previous night's sleep, "
        "and physical activity are important."
    )

    is_valid, reason = validate_generated_explanation(
        content, prediction, top_factors
    )

    assert is_valid is False
    assert reason == "Prediction value is incorrect."