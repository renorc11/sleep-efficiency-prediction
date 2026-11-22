"""Phase 5: whole-API integration with the real deployed models.

Every endpoint is exercised through one app lifespan with an isolated
in-memory database. Retrieval and BioMistral are faked so the test is
deterministic and offline. Run from the project root.
"""

import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.main as main_module
import app.services.chat_service as chat_service
import app.services.personalized_recommendation_service as prs
from app.core.config import BACKEND_DIR, resolve_database_url
from app.db.models import PredictionRecord
from app.db.session import get_db
from app.main import app
from app.schemas.prediction import PredictionResponse
from app.services.recommendation_context_service import RECOMMENDATION_TOPICS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = json.loads((PROJECT_ROOT / "ml/data/knowledge/sleep_knowledge_dataset_v0.3.1.json").read_text())
CHUNKS = {c["chunk_id"]: c for c in DATASET["knowledge_chunks"]}
REGISTRY = {s["source_id"]: s for s in DATASET["source_registry"]}
PAYLOAD = json.loads((PROJECT_ROOT / "ml/data/raw/example_prediction_payload.json").read_text())
QUERY_TO_CHUNK = {t["query"]: t["chunk_id"] for t in RECOMMENDATION_TOPICS.values() if t.get("query")}


class FakeKnowledge:
    def retrieve(self, question, top_k=3):
        chunk_id = QUERY_TO_CHUNK.get(question, "CAFF-001")
        chunk = CHUNKS[chunk_id]
        return {
            "dataset_version": "0.3.1",
            "chunks": [{**chunk, "similarity": 0.8}],
            "sources": [{"source_id": s, "title": REGISTRY[s]["title"]} for s in chunk["source_ids"]],
        }


class DownClient:
    """BioMistral unreachable for the sync recommendation client."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def post(self, url, json):
        raise httpx.ConnectError("BioMistral down")


class ChatClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, url, json):
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": "Caffeine late in the day can disrupt sleep (SRC-005)."}}]},
        )


@pytest.fixture(scope="module")
def api():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Session = sessionmaker(bind=engine)

    def override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    with patch.object(main_module, "engine", engine), \
            patch.object(main_module, "create_knowledge_service", return_value=FakeKnowledge()), \
            patch.object(prs.httpx, "Client", DownClient), \
            patch.object(chat_service.httpx, "AsyncClient", ChatClient):
        with TestClient(app) as client:
            yield client, Session
    app.dependency_overrides.clear()
    engine.dispose()


def test_health_reports_components(api):
    client, _ = api
    body = client.get("/api/v1/health").json()
    assert body == {"status": "ok", "knowledge_service": "available", "models": "loaded"}
    assert app.state.models_loaded is True


def test_predict_history_round_trip(api):
    client, Session = api
    responses = []
    for stress in (3, 8):
        payload = {**PAYLOAD, "lifestyle": {**PAYLOAD["lifestyle"], "stress_level": stress}}
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200, response.text
        body = PredictionResponse(**response.json())
        assert body.personalized_recommendations is not None
        assert body.personalized_recommendations.generation_status == "fallback"  # BioMistral down
        assert body.prediction_diagnostics["raw_outputs"]["rf"] == pytest.approx(body.model_outputs["rf"], abs=0.005)
        responses.append(body)

    history = client.get("/api/v1/history").json()
    assert [round(h["predicted_sleep_efficiency"], 2) for h in history[:2]] == [
        responses[1].predicted_sleep_efficiency,
        responses[0].predicted_sleep_efficiency,
    ]
    assert len(client.get("/api/v1/history", params={"limit": 1}).json()) == 1
    for bad in (0, -5, 500):
        assert client.get("/api/v1/history", params={"limit": bad}).status_code == 422

    with Session() as db:
        stored = db.query(PredictionRecord).order_by(PredictionRecord.id).all()
    assert len(stored) >= 2
    assert stored[-1].input_payload["lifestyle"]["stress_level"] == 8
    assert set(stored[-1].model_outputs) == {"rf", "xgb", "hybrid", "ensemble"}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["lifestyle"].pop("stress_level"),
        lambda p: p["lifestyle"].update(stress_level=99),
        lambda p: p.update(screen_activity_sequence=[2] * 288),
        lambda p: p.update(screen_activity_sequence=[0] * 100),
        lambda p: p.update(sleep_time_hint="25:00"),
    ],
)
def test_predict_rejects_invalid_payloads(api, mutate):
    client, _ = api
    payload = json.loads(json.dumps(PAYLOAD))
    mutate(payload)
    assert client.post("/api/v1/predict", json=payload).status_code == 422


def test_explain_endpoints(api):
    client, _ = api
    global_body = client.get("/api/v1/explain/global").json()
    assert 1 <= len(global_body["global_feature_importance"]) <= 15
    assert global_body["explanation"]["method"].startswith("TreeSHAP")

    local = client.post("/api/v1/explain/local", json=PAYLOAD)
    assert local.status_code == 200
    body = local.json()
    assert body["model_explanations"]["rf"]["method"] == "TreeSHAP"
    assert body["model_explanations"]["hybrid"]["method"] == "IntegratedGradients"
    assert len(body["model_explanations"]["hybrid"]["temporal"]["positions"]) == 288
    assert body["prediction_diagnostics"]["ensemble_policy"]["weights"] == {"rf": 0.2, "xgb": 0.3, "hybrid": 0.5}

    body = client.get("/api/v1/explain/model-metrics").json()
    metrics = body["metrics_summary"]
    assert {"rf", "xgb", "hybrid"} <= set(metrics)
    regression = body["metric_groups"]["regression"]["metrics"]
    threshold_derived = body["metric_groups"]["threshold_derived"]["metrics"]
    assert regression == ["mae", "rmse", "r2"]
    assert threshold_derived == ["accuracy", "precision", "recall", "f1"]
    assert not set(regression) & set(threshold_derived)
    assert body["metric_groups"]["threshold_derived"]["threshold"] == 85.0
    for values in metrics.values():
        assert "roc_auc" not in values  # not a valid metric for this regression model
        assert set(regression) | set(threshold_derived) <= set(values)
    assert "roc_auc" in body["withheld_metrics"]


def test_model_metrics_leave_artifact_metadata_unchanged(api):
    from app.services.model_service import get_model_bundle

    client, _ = api
    client.get("/api/v1/explain/model-metrics")
    # The stored artifact metadata still holds the original value; only the API withholds it.
    assert "roc_auc" in get_model_bundle().meta["metrics_summary"]["rf"]


def test_chat_endpoint(api):
    client, _ = api
    response = client.post("/api/v1/chat", json={"message": "Does evening caffeine affect sleep?"})
    assert response.status_code == 200
    assert response.json() == {"reply": "Caffeine late in the day can disrupt sleep (SRC-005)."}
    assert client.post("/api/v1/chat", json={"message": ""}).status_code == 422


@pytest.mark.parametrize(
    "origin, allowed",
    [
        ("http://localhost:3000", True),
        ("http://localhost:3001", True),  # Next.js fallback port when 3000 is busy
        ("http://127.0.0.1:3002", True),
        ("http://evil.example.com", False),
        ("http://localhost.evil.com", False),
    ],
)
def test_cors_preflight_for_local_frontend_ports(api, origin, allowed):
    client, _ = api
    response = client.options(
        "/api/v1/predict",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert (response.status_code == 200) is allowed
    assert (response.headers.get("access-control-allow-origin") == origin) is allowed


def test_openapi_documents_every_route(api):
    client, _ = api
    paths = set(client.get("/openapi.json").json()["paths"])
    assert {
        "/api/v1/health", "/api/v1/predict", "/api/v1/history", "/api/v1/explain/global",
        "/api/v1/explain/local", "/api/v1/explain/model-metrics", "/api/v1/chat",
    } <= paths


@pytest.mark.parametrize(
    "url, expected",
    [
        ("sqlite:///./sleep_app.db", f"sqlite:///{(BACKEND_DIR / 'sleep_app.db').resolve()}"),
        ("sqlite:///data/app.db", f"sqlite:///{(BACKEND_DIR / 'data/app.db').resolve()}"),
        ("sqlite:////tmp/abs.db", "sqlite:////tmp/abs.db"),
        ("sqlite://", "sqlite://"),
        ("postgresql://user@host/db", "postgresql://user@host/db"),
    ],
)
def test_relative_sqlite_paths_anchor_to_backend(url, expected):
    assert resolve_database_url(url) == expected
