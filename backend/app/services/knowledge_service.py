
import json
import os
import re
import sys

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_VERSION = os.getenv(
    "KNOWLEDGE_DATASET_VERSION",
    "0.3.1",
)

DATASET_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "knowledge"
    / f"sleep_knowledge_dataset_v{DATASET_VERSION}.json"
)

DEFAULT_TOP_K = 3
DEFAULT_MIN_SIMILARITY = float(
    os.getenv("KNOWLEDGE_MIN_SIMILARITY", "0.65")
)

if not 0.0 <= DEFAULT_MIN_SIMILARITY <= 1.0:
    raise ValueError(
        "KNOWLEDGE_MIN_SIMILARITY must be between 0.0 and 1.0."
    )

def parse_metadata_list(value: Any) -> list[str]:
    """
    Normalize list-like ChromaDB metadata.

    ChromaDB stores list metadata as JSON strings in this project.
    """
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        try:
            parsed = json.loads(value)

            if isinstance(parsed, list):
                return [str(item) for item in parsed]

        except json.JSONDecodeError:
            pass

        return [value]

    raise ValueError("Unsupported list metadata format.")


def normalize_source_url(value: str | None) -> str | None:
    """Extract a plain URL from a Markdown link if necessary."""
    if not value:
        return None

    value = value.strip()

    match = re.fullmatch(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        value,
    )

    if match:
        return match.group(2)

    return value


class SleepKnowledgeService:
    """
    Retrieves sleep knowledge and resolves its bibliographic sources.

    The retriever is injected so it can be initialized once during
    application startup and mocked during automated tests.
    """

    def __init__(
    self,
    retriever: Any,
    version: str = DATASET_VERSION,
    dataset_path: Path = DATASET_PATH,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ) -> None:

        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between 0.0 and 1.0.")
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Knowledge dataset not found: {dataset_path}"
            )

        with dataset_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            dataset = json.load(file)

        if dataset.get("dataset_version") != version:
            raise ValueError(
                "Knowledge dataset version mismatch. "
                f"Expected {version}, "
                f"found {dataset.get('dataset_version')}."
            )

        registry = dataset.get("source_registry")

        if not isinstance(registry, list):
            raise ValueError(
                "Knowledge dataset has an invalid source registry."
            )

        self.retriever = retriever
        self.min_similarity = min_similarity
        self.version = version

        self.source_registry = {
            source["source_id"]: source
            for source in registry
            if isinstance(source, dict)
            and source.get("source_id")
        }

        if not self.source_registry:
            raise ValueError(
                "Knowledge source registry is empty or invalid."
            )

    def _resolve_sources(
        self,
        source_ids: list[str],
    ) -> list[dict[str, Any]]:

        resolved = []

        for source_id in source_ids:

            source = self.source_registry.get(source_id)

            if source is None:
                raise RuntimeError(
                    f"Unknown knowledge source ID: {source_id}"
                )

            resolved.append({
                "source_id": source_id,
                "title": source.get("title"),
                "authors": source.get("authors", []),
                "organization": source.get("organization"),
                "year": source.get("year"),
                "publication_type": source.get("publication_type"),
                "journal": source.get("journal"),
                "doi": source.get("doi"),
                "url": normalize_source_url(source.get("url")),
            })

        return resolved

    def retrieve(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> dict[str, Any]:

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not 1 <= top_k <= 17:
            raise ValueError("top_k must be between 1 and 17.")

        results = self.retriever.search(
            question=question.strip(),
            top_k=top_k,
        )

        chunks = []
        sources_by_id = {}

        for result in results:
            similarity = result.get("similarity")

            if (similarity is None or not isinstance(similarity, (int, float))
            or not 0.0 <= similarity <= 1.0
            or similarity < self.min_similarity
            ):
                continue

            metadata = result.get("metadata") or {}

            source_ids = parse_metadata_list(
                metadata.get("source_ids")
            )

            resolved_sources = self._resolve_sources(
                source_ids
            )

            for source in resolved_sources:
                sources_by_id[source["source_id"]] = source

            chunks.append({
                "chunk_id": result["id"],
                "topic": result.get("topic", ""),
                "category": result.get("category", ""),
                "content": result.get("text", ""),
                "similarity": result.get("similarity"),
                "evidence_type": metadata.get("evidence_type"),
                "evidence_strength": metadata.get("evidence_strength"),
                "population_context": metadata.get("population_context"),
                "practical_implication": metadata.get("practical_implication"),
                "limitations": parse_metadata_list(
                    metadata.get("limitations")
                ),
                "source_ids": source_ids,
            })

        return {
            "dataset_version": self.version,
            "chunks": chunks,
            "sources": list(sources_by_id.values()),
        }


def ensure_project_root_importable() -> None:
    """
    Make the top-level ``ml`` package importable.

    The documented startup command runs from ``backend/``, where the project
    root is not on ``sys.path``. It is appended (not prepended) so it can never
    shadow the backend's own packages.
    """
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.append(project_root)


def create_knowledge_service(
    version: str = DATASET_VERSION,
) -> SleepKnowledgeService:
    """
    Create the real retrieval service.

    Importing the retriever here keeps the dependency lazy and allows
    tests to exercise source resolution without loading the ML model.

    """
    ensure_project_root_importable()

    from ml.scripts.search_knowledge import SleepKnowledgeRetriever

    retriever = SleepKnowledgeRetriever(version=version)

    return SleepKnowledgeService(
        retriever=retriever,
        version=version,
    )
