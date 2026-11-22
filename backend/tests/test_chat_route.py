from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app


def test_chat_route_passes_knowledge_service():
    fake_knowledge_service = MagicMock()

    with patch(
        "app.main.create_knowledge_service",
        return_value=fake_knowledge_service,
    ):
        with patch(
            "app.api.routes.chat.generate_chat_response",
            new_callable=AsyncMock,
            return_value="Caffeine can affect sleep.",
        ) as mock_generate:

            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/chat",
                    json={"message": "How does caffeine affect sleep?"},
                )

            assert response.status_code == 200
            assert response.json() == {
                "reply": "Caffeine can affect sleep."
            }

            mock_generate.assert_awaited_once_with(
                "How does caffeine affect sleep?",
                fake_knowledge_service,
            )


def test_chat_route_rejects_empty_message():
    # Validation needs no retrieval; without this patch app startup built the
    # real knowledge service and opened the vector store.
    with patch("app.main.create_knowledge_service", return_value=MagicMock()), TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={"message": "   "},
        )

    assert response.status_code == 422


def test_chat_route_returns_503_on_service_failure():
    fake_knowledge_service = MagicMock()

    with patch(
        "app.main.create_knowledge_service",
        return_value=fake_knowledge_service,
    ):
        with patch(
            "app.api.routes.chat.generate_chat_response",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Knowledge retrieval failed."),
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/chat",
                    json={"message": "How can I improve my sleep?"},
                )

    assert response.status_code == 503
    assert response.json()["detail"] == "Knowledge retrieval failed."