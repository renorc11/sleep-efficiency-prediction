
import json

from pathlib import Path

import pytest

from app.services.knowledge_service import (
    SleepKnowledgeService,
    parse_metadata_list,
    normalize_source_url,
)


class FakeRetriever:

    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def search(self, question, top_k):
        self.calls.append({
            "question": question,
            "top_k": top_k,
        })
        return self.results


@pytest.fixture
def dataset_file(tmp_path):

    dataset = {
        "dataset_version": "0.3.1",
        "source_registry": [
            {
                "source_id": "SRC-001",
                "title": "Sleep research reference",
                "authors": ["Author One"],
                "organization": "Test Organization",
                "year": 2021,
                "publication_type": "research_article",
                "journal": "Sleep Journal",
                "doi": "10.1234/example",
                "url": (
                    "[https://example.com/source]"
                    "(https://example.com/source)"
                ),
            }
        ],
    }

    path = tmp_path / "test_knowledge.json"

    path.write_text(
        json.dumps(dataset),
        encoding="utf-8",
    )

    return path


@pytest.fixture
def fake_retriever():

    return FakeRetriever([
        {
            "id": "SLEEP-001",
            "topic": "Healthy sleep",
            "category": "sleep_fundamentals",
            "text": "Healthy sleep includes adequate duration.",
            "similarity": 0.85,
            "metadata": {
                "evidence_type": "research_article",
                "evidence_strength": "moderate",
                "population_context": "Adults",
                "practical_implication": "Consider sleep duration.",
                "limitations": '["Not a diagnostic tool."]',
                "source_ids": '["SRC-001"]',
            },
        }
    ])


@pytest.fixture
def knowledge_service(fake_retriever, dataset_file):

    return SleepKnowledgeService(
        retriever=fake_retriever,
        version="0.3.1",
        dataset_path=dataset_file,
    )


def test_metadata_list_parsing():

    assert parse_metadata_list(
        '["SRC-001", "SRC-002"]'
    ) == ["SRC-001", "SRC-002"]

    assert parse_metadata_list(
        ["SRC-001"]
    ) == ["SRC-001"]

    assert parse_metadata_list(None) == []

    assert parse_metadata_list("") == []


def test_source_url_normalization():

    result = normalize_source_url(
        "[https://example.com/source]"
        "(https://example.com/source)"
    )

    assert result == "https://example.com/source"


def test_knowledge_service_initialization(knowledge_service):

    assert knowledge_service.version == "0.3.1"

    assert "SRC-001" in knowledge_service.source_registry


def test_dataset_version_mismatch(fake_retriever, dataset_file):

    with pytest.raises(ValueError, match="version mismatch"):

        SleepKnowledgeService(
            retriever=fake_retriever,
            version="0.3.0",
            dataset_path=dataset_file,
        )


def test_missing_dataset_file(fake_retriever, tmp_path):

    with pytest.raises(FileNotFoundError):

        SleepKnowledgeService(
            retriever=fake_retriever,
            version="0.3.1",
            dataset_path=tmp_path / "missing.json",
        )


def test_retrieve_returns_chunks_and_sources(knowledge_service):

    result = knowledge_service.retrieve(
        "How does sleep duration affect sleep health?"
    )

    assert result["dataset_version"] == "0.3.1"

    assert len(result["chunks"]) == 1

    chunk = result["chunks"][0]

    assert chunk["chunk_id"] == "SLEEP-001"

    assert chunk["content"] == (
        "Healthy sleep includes adequate duration."
    )

    assert chunk["source_ids"] == ["SRC-001"]

    assert len(result["sources"]) == 1

    source = result["sources"][0]

    assert source["source_id"] == "SRC-001"

    assert source["title"] == "Sleep research reference"

    assert source["url"] == "https://example.com/source"


def test_retriever_receives_question_and_top_k(
    knowledge_service,
    fake_retriever,
):

    knowledge_service.retrieve(
        "What is sleep efficiency?",
        top_k=2,
    )

    assert fake_retriever.calls == [
        {
            "question": "What is sleep efficiency?",
            "top_k": 2,
        }
    ]


def test_empty_question_is_rejected(knowledge_service):

    with pytest.raises(ValueError, match="cannot be empty"):

        knowledge_service.retrieve("   ")


@pytest.mark.parametrize("top_k", [0, -1, 18])

def test_invalid_top_k_is_rejected(knowledge_service, top_k):

    with pytest.raises(ValueError, match="top_k"):

        knowledge_service.retrieve(
            "Sleep question",
            top_k=top_k,
        )


def test_unknown_source_id_is_rejected(
    fake_retriever,
    dataset_file,
):

    fake_retriever.results[0]["metadata"]["source_ids"] = (
        '["SRC-UNKNOWN"]'
    )

    service = SleepKnowledgeService(
        retriever=fake_retriever,
        version="0.3.1",
        dataset_path=dataset_file,
    )

    with pytest.raises(
        RuntimeError,
        match="Unknown knowledge source ID",
    ):

        service.retrieve("Sleep question")


def test_duplicate_sources_are_deduplicated(
    fake_retriever,
    knowledge_service,
):

    duplicate = dict(fake_retriever.results[0])

    duplicate["id"] = "SLEEP-002"

    fake_retriever.results.append(duplicate)

    result = knowledge_service.retrieve("Sleep question")

    assert len(result["chunks"]) == 2

    assert len(result["sources"]) == 1

def test_retrieve_rejects_low_similarity_results(
    fake_retriever,
    knowledge_service,
):
    fake_retriever.results[0]["similarity"] = 0.569

    result = knowledge_service.retrieve(
        "Explain compound annual growth rate."
    )

    assert result["chunks"] == []
    assert result["sources"] == []


def test_retrieve_accepts_similarity_above_threshold(
    fake_retriever,
    knowledge_service,
):
    fake_retriever.results[0]["similarity"] = 0.85

    result = knowledge_service.retrieve(
        "How does sleep duration affect sleep health?"
    )

    assert len(result["chunks"]) == 1
    assert len(result["sources"]) == 1


def test_retrieve_accepts_similarity_at_threshold(
    fake_retriever,
    knowledge_service,
):
    fake_retriever.results[0]["similarity"] = 0.65

    result = knowledge_service.retrieve(
        "What is sleep efficiency?"
    )

    assert len(result["chunks"]) == 1


def test_retrieve_filters_low_similarity_before_source_resolution(
    fake_retriever,
    knowledge_service,
):
    fake_retriever.results[0]["similarity"] = 0.50
    fake_retriever.results[0]["metadata"]["source_ids"] = (
        '["SRC-UNKNOWN"]'
    )

    result = knowledge_service.retrieve("Unrelated question")

    assert result["chunks"] == []
    assert result["sources"] == []