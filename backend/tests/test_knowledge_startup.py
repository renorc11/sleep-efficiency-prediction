import os

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app


def test_knowledge_service_initializes_at_startup():

    fake_service = MagicMock()

    with patch(
        "app.main.create_knowledge_service",
        return_value=fake_service,
    ) as create_service:

        with TestClient(app) as client:

            assert client.get(
                "/api/v1/health"
            ).status_code == 200

            assert app.state.knowledge_service is fake_service

        create_service.assert_called_once_with()


def test_knowledge_service_initializes_once_per_startup():

    fake_service = MagicMock()

    with patch(
        "app.main.create_knowledge_service",
        return_value=fake_service,
    ) as create_service:

        with TestClient(app):

            create_service.assert_called_once()

            assert app.state.knowledge_service is fake_service


def test_startup_fails_if_knowledge_service_initialization_fails():
    # Fail-fast is now opt-in; the default starts in degraded mode
    # (see test_phase3_rag_llm.py).
    with patch.dict(
        os.environ,
        {"KNOWLEDGE_SERVICE_REQUIRED": "true"},
    ), patch(
        "app.main.create_knowledge_service",
        side_effect=RuntimeError(
            "Knowledge service initialization failed."
        ),
    ):

        try:
            with TestClient(app):
                assert False, "Application startup should have failed."

        except RuntimeError as exc:
            assert "Knowledge service initialization failed." in str(exc)
