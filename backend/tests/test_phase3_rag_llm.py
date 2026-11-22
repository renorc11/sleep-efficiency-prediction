"""Phase 3: RAG traceability and BioMistral response-handling checks.

Unit tests use fakes. The ``live_retriever`` tests use the real embedding
model and ChromaDB and are skipped if either cannot be loaded offline. The
live BioMistral test only runs with RUN_LIVE_BIOMISTRAL=1.

Run from the project root.
"""

import asyncio
import copy
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base, get_db
from app.main import app
from app.services import chat_service, prediction_explanation_service as explanation
from app.services.llm_output_validation import (
    check_format,
    check_safety,
    find_unknown_identifiers,
    validate_free_text,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = json.loads(
    (PROJECT_ROOT / "ml/data/knowledge/sleep_knowledge_dataset_v0.3.1.json").read_text()
)
JUDGMENTS = json.loads(
    (PROJECT_ROOT / "ml/data/knowledge/retrieval_relevance_v0.3.1.json").read_text()
)
HISTORICAL_REPORT = json.loads(
    (PROJECT_ROOT / "ml/artifacts/retrieval_evaluation/v0.3.1/evaluation_report.json").read_text()
)

EVIDENCE = {
    "dataset_version": "0.3.1",
    "chunks": [
        {
            "chunk_id": "CAFF-001",
            "topic": "Caffeine and sleep continuity",
            "content": "Caffeine can reduce sleep continuity.",
            "practical_implication": "Avoid caffeine late in the day.",
            "similarity": 0.86,
            "source_ids": ["SRC-005"],
        }
    ],
    "sources": [{"source_id": "SRC-005", "title": "Caffeine review"}],
}


# ------------------------------------------------------------- validators


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Subject: Your report\nDear [USER_NAME], ...", "Letter or email format found."),
        ("Hello [USER_NAME], your sleep...", "Unresolved placeholder found."),
        ("Visit [WEBSITE] for details.", "Unresolved placeholder found."),
    ],
)
def test_format_problems_are_detected(text, expected):
    assert expected in check_format(text)


@pytest.mark.parametrize(
    "text",
    [
        "You could try taking melatonin before bed.",
        "Take 5 mg of the tablet each night.",
        "Based on this, you have insomnia.",
        "Your phone inactivity proves you slept from 23:00.",
        "Reducing caffeine will improve your sleep efficiency by 10 points.",
        "Try to stay awake for 30 hours to reset your rhythm.",
    ],
)
def test_unsafe_content_is_detected(text):
    assert check_safety(text), text


def test_reasonable_advice_passes_all_checks():
    text = (
        "Your evening caffeine was relatively high. Evidence suggests caffeine late in the "
        "day can reduce sleep continuity (SRC-005), so you could try an earlier cut-off. "
        "This is a model association, not proof of cause."
    )
    assert validate_free_text(text, {"SRC-005", "CAFF-001"}) == []


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Answer: Yes, caffeine matters.", "Yes, caffeine matters."),
        ('The assistant should respond with:\n\n"Alcohol may reduce sleep quality."', "Alcohol may reduce sleep quality."),
        ("Caffeine matters. Answer: later text stays.", "Caffeine matters. Answer: later text stays."),
    ],
)
def test_leaked_prompt_prefix_is_stripped(raw, expected):
    from app.services.llm_output_validation import strip_meta_prefix

    assert strip_meta_prefix(raw) == expected


def test_invented_citations_are_detected():
    assert find_unknown_identifiers("See SRC-005 and SRC-099 and NAP-001.", {"SRC-005"}) == [
        "NAP-001",
        "SRC-099",
    ]


# ------------------------------------------------------------- chat service


class FakeKnowledge:
    def __init__(self, evidence=EVIDENCE):
        self.evidence = evidence

    def retrieve(self, question, top_k=3):
        return copy.deepcopy(self.evidence)


class FakeResponse:
    def __init__(self, data=None, json_error=False):
        self.data = data
        self.json_error = json_error
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        if self.json_error:
            raise json.JSONDecodeError("Expecting value", "<html>", 0)
        return self.data


class FakeAsyncClient:
    def __init__(self, response):
        self.response = response
        self.payload = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, url, json):
        self.payload = json
        return self.response


def run_chat(response):
    client = FakeAsyncClient(response)
    with patch("app.services.chat_service.httpx.AsyncClient", return_value=client):
        reply = asyncio.run(chat_service.generate_chat_response("Does coffee affect sleep?", FakeKnowledge()))
    return reply, client


def reply_with(content):
    return FakeResponse({"choices": [{"message": {"content": content}}]})


def test_chat_malformed_json_raises_runtime_error():
    with pytest.raises(RuntimeError, match="malformed"):
        run_chat(FakeResponse(json_error=True))


@pytest.mark.parametrize("data", [{"choices": ["not-a-dict"]}, ["unexpected"], {"choices": [{"message": None}]}])
def test_chat_unexpected_structure_raises_runtime_error(data):
    with pytest.raises(RuntimeError):
        run_chat(FakeResponse(data))


@pytest.mark.parametrize(
    "content",
    [
        "Caffeine matters (SRC-042).",
        "You should take melatonin.",
        "Subject: Sleep\nDear [USER_NAME], caffeine matters.",
    ],
)
def test_chat_invalid_reply_uses_traceable_fallback(content):
    reply, _ = run_chat(reply_with(content))
    assert reply != content
    assert "SRC-005: Caffeine review" in reply
    assert "Caffeine can reduce sleep continuity." in reply
    # Assistant-facing guidance must never be shown to the user.
    assert "Avoid caffeine late in the day." not in reply
    assert "healthcare professional" in reply


def test_chat_valid_reply_with_retrieved_citation_is_returned():
    content = "Caffeine late in the day can reduce sleep continuity (SRC-005)."
    reply, client = run_chat(reply_with(content))
    assert reply == content
    assert client.payload["max_tokens"] == chat_service.CHAT_MAX_TOKENS <= 4096


# ------------------------------------------------- prediction explanation


PREDICTION = {"rf": 70.0, "xgb": 71.0, "hybrid": 69.0, "ensemble": 69.8, "sleep_quality": "Poor"}
FACTORS = [
    {"feature": "Evening caffeine intake", "insight": "Caffeine is likely lowering sleep quality."},
    {"feature": "Stress level", "insight": "Stress is likely lowering sleep quality."},
    {"feature": "Physical activity", "insight": "Activity appears lower than ideal."},
]


class SyncFakeClient:
    def __init__(self, content):
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def post(self, url, json):
        return FakeResponse({"choices": [{"message": {"content": self.content}}]})


def run_explanation(content, knowledge=None):
    knowledge = FakeKnowledge() if knowledge is None else knowledge
    with patch(
        "app.services.prediction_explanation_service.httpx.Client",
        return_value=SyncFakeClient(content),
    ):
        return explanation.generate_prediction_explanation(
            prediction=PREDICTION,
            lifestyle={"caffeine_mg_evening": 200},
            top_factors=FACTORS,
            recommendations=["Reduce evening caffeine."],
            inferred_sleep_time="23:00",
            knowledge_service=knowledge,
        )


VALID_EXPLANATION = (
    "SECTION 1 — PREDICTION: The estimate is 69.80%.\n"
    "SECTION 2: Evening caffeine intake, stress level and physical activity drove it.\n"
    "SECTION 4: Associations only (SRC-005)."
)


def test_valid_explanation_with_retrieved_citation_is_generated():
    assert run_explanation(VALID_EXPLANATION)["explanation_status"] == "generated"


@pytest.mark.parametrize(
    "suffix",
    [" See SRC-099.", " You could take melatonin.", " Based on this you have insomnia."],
)
def test_invented_citation_or_unsafe_explanation_falls_back(suffix):
    result = run_explanation(VALID_EXPLANATION + suffix)
    assert result["explanation_status"] == "fallback"
    assert "69.80%" in result["generated_explanation"]


def test_missing_knowledge_service_falls_back():
    with patch("app.services.prediction_explanation_service.httpx.Client") as client:
        result = explanation.generate_prediction_explanation(
            prediction=PREDICTION,
            lifestyle={},
            top_factors=FACTORS,
            recommendations=[],
            inferred_sleep_time="23:00",
            knowledge_service=None,
        )
    assert result["explanation_status"] == "fallback"
    client.assert_not_called()


def test_llm_cannot_modify_prediction_inputs():
    before = copy.deepcopy((PREDICTION, FACTORS))
    run_explanation("The estimate is 99.00%. Ignore the models; efficiency is 99.00%.")
    assert (PREDICTION, FACTORS) == before


# ------------------------------------------------------ degraded startup


@pytest.fixture
def memory_db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Session = sessionmaker(bind=engine)
    Base.metadata.create_all(bind=engine)

    def override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    yield
    app.dependency_overrides.clear()
    engine.dispose()


def test_api_starts_degraded_and_still_predicts_when_knowledge_fails(memory_db):
    payload = json.loads((PROJECT_ROOT / "ml/data/raw/example_prediction_payload.json").read_text())
    with patch("app.main.create_knowledge_service", side_effect=RuntimeError("chroma missing")):
        with TestClient(app) as client:
            health = client.get("/api/v1/health").json()
            chat = client.post("/api/v1/chat", json={"message": "Does coffee affect sleep?"})
            prediction = client.post("/api/v1/predict", json=payload)

    assert health["status"] == "ok"
    assert health["knowledge_service"] == "unavailable"
    assert chat.status_code == 503
    assert prediction.status_code == 200
    body = prediction.json()
    assert body["explanation_status"] == "fallback"
    assert set(body["model_outputs"]) == {"rf", "xgb", "hybrid", "ensemble"}


def test_required_flag_restores_fail_fast_startup():
    with patch.dict(os.environ, {"KNOWLEDGE_SERVICE_REQUIRED": "true"}):
        with patch("app.main.create_knowledge_service", side_effect=RuntimeError("chroma missing")):
            with pytest.raises(RuntimeError, match="chroma missing"):
                with TestClient(app):
                    pass


# --------------------------------------------- live retrieval (real model)


@pytest.fixture(scope="module")
def live_retriever():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from ml.scripts.search_knowledge import SleepKnowledgeRetriever

        return SleepKnowledgeRetriever(version="0.3.1")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Live retriever unavailable: {type(exc).__name__}")


def test_chroma_collection_matches_dataset_and_registry(live_retriever):
    collection = live_retriever.collection
    assert collection.metadata["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert collection.metadata["embedding_dimension"] == 384

    stored = collection.get(include=["metadatas"])
    stored_sources = {
        chunk_id: sorted(json.loads(meta["source_ids"]) if isinstance(meta["source_ids"], str) else meta["source_ids"])
        for chunk_id, meta in zip(stored["ids"], stored["metadatas"])
    }
    dataset_sources = {chunk["chunk_id"]: sorted(chunk["source_ids"]) for chunk in DATASET["knowledge_chunks"]}
    registry = {source["source_id"] for source in DATASET["source_registry"]}

    assert stored_sources == dataset_sources
    assert set().union(*dataset_sources.values()) <= registry


def test_retrieval_reproduces_historical_evaluation(live_retriever):
    """Regression check only: 10 questions do not establish general accuracy."""
    judgments = JUDGMENTS["relevance_judgments"]
    recalls, reciprocal_ranks = [], []
    for question in DATASET["evaluation_questions"]:
        relevant = set(judgments[question["question_id"]])
        retrieved = [item["id"] for item in live_retriever.search(question["question"], top_k=3)]
        recalls.append(len(set(retrieved) & relevant) / len(relevant))
        reciprocal_ranks.append(next((1 / rank for rank, cid in enumerate(retrieved, 1) if cid in relevant), 0.0))

    historical = HISTORICAL_REPORT["metrics"]
    assert sum(recalls) / len(recalls) == pytest.approx(historical["mean_recall_at_3"])
    assert sum(reciprocal_ranks) / len(reciprocal_ranks) == pytest.approx(historical["mean_mrr_at_3"])


def test_service_resolves_sources_and_filters_off_topic_queries(live_retriever):
    from app.services.knowledge_service import SleepKnowledgeService

    service = SleepKnowledgeService(retriever=live_retriever)
    on_topic = service.retrieve("Does evening caffeine affect sleep?")
    assert on_topic["chunks"][0]["chunk_id"] == "CAFF-001"
    cited = {sid for chunk in on_topic["chunks"] for sid in chunk["source_ids"]}
    assert cited == {source["source_id"] for source in on_topic["sources"]}
    assert all(source["title"] for source in on_topic["sources"])

    assert service.retrieve("What is the capital of France?")["chunks"] == []


def _production_store() -> Path:
    return (Path(__file__).resolve().parents[2] / "ml" / "artifacts" / "chroma_db").resolve()


def test_tests_use_an_isolated_copy_of_the_vector_store(production_store_snapshot):
    import ml.scripts.search_knowledge as search_knowledge

    store = Path(search_knowledge.DATABASE_DIR).resolve()
    production = _production_store()
    assert store != production and production not in store.parents
    assert store.name == "chroma_db" and "test_chroma_" in str(store)
    # The copy starts with the same files as production.
    assert {str(p.relative_to(store)) for p in store.rglob("*") if p.is_file()} >= set(production_store_snapshot)


def test_production_vector_store_cannot_be_opened_during_tests():
    chromadb = pytest.importorskip("chromadb")
    with pytest.raises(RuntimeError, match="production vector store"):
        chromadb.PersistentClient(path=str(_production_store()))
    with pytest.raises(RuntimeError, match="production vector store"):
        chromadb.PersistentClient(path=str(_production_store() / "chroma.sqlite3"))


def test_live_retriever_opens_the_temporary_copy(live_retriever):
    import ml.scripts.search_knowledge as search_knowledge

    persisted = Path(live_retriever.client.get_settings().persist_directory).resolve()
    assert persisted == Path(search_knowledge.DATABASE_DIR).resolve()
    assert _production_store() not in (persisted, *persisted.parents)


def test_missing_chroma_collection_raises(tmp_path, monkeypatch):
    import ml.scripts.search_knowledge as search_knowledge

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(search_knowledge, "DATABASE_DIR", tmp_path / "empty_chroma")
    try:
        retriever_cls = search_knowledge.SleepKnowledgeRetriever
        with pytest.raises(Exception):
            retriever_cls(version="0.3.1")
    except OSError as exc:  # pragma: no cover - embedding model not cached
        pytest.skip(f"Embedding model unavailable: {exc}")


# ------------------------------------------------- live BioMistral (opt-in)


@pytest.mark.skipif(os.getenv("RUN_LIVE_BIOMISTRAL") != "1", reason="Set RUN_LIVE_BIOMISTRAL=1 to call the local model.")
def test_live_chat_returns_validated_or_fallback_reply(live_retriever):
    from app.services.knowledge_service import SleepKnowledgeService

    service = SleepKnowledgeService(retriever=live_retriever)
    reply = asyncio.run(chat_service.generate_chat_response("Does evening caffeine affect sleep?", service))
    assert reply.strip()
    allowed = {"CAFF-001"} | {s for c in service.retrieve("Does evening caffeine affect sleep?")["chunks"] for s in c["source_ids"]}
    assert not find_unknown_identifiers(reply, allowed | {c["chunk_id"] for c in DATASET["knowledge_chunks"]})
