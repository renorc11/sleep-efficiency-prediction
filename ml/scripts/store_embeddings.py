import os
import json
import sys
from pathlib import Path

import chromadb
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_VERSION = os.getenv("KNOWLEDGE_DATASET_VERSION", "0.3.1")

DATASET_PATH = (
    PROJECT_ROOT
    / f"ml/data/knowledge/sleep_knowledge_v{DATASET_VERSION}.jsonl"
)

EMBEDDINGS_DIR = (
    PROJECT_ROOT
    / f"ml/artifacts/knowledge_embeddings/v{DATASET_VERSION}"
)

DATABASE_DIR = PROJECT_ROOT / "ml/artifacts/chroma_db"

COLLECTION_NAME = (
    f"sleep_knowledge_v{DATASET_VERSION.replace('.', '_')}"
)

EXPECTED_RECORD_COUNT = 17
EXPECTED_DIMENSION = 384
EXPECTED_DATASET_VERSION = DATASET_VERSION

EXPECTED_MODEL = "BAAI/bge-small-en-v1.5"


def load_inputs():
    embedding_file = EMBEDDINGS_DIR / "embeddings.npz"
    metadata_file = EMBEDDINGS_DIR / "metadata.json"
    report_file = EMBEDDINGS_DIR / "generation_report.json"

    for path in (DATASET_PATH, embedding_file, metadata_file, report_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    records = []
    with DATASET_PATH.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at line {line_number}"
                    ) from exc

    archive = np.load(embedding_file, allow_pickle=False)
    ids = archive["ids"].astype(str).tolist()
    vectors = np.asarray(archive["embeddings"], dtype=np.float32)

    metadata_records = json.loads(
        metadata_file.read_text(encoding="utf-8")
    )

    report = json.loads(report_file.read_text(encoding="utf-8"))

    if report.get("dataset_version") != EXPECTED_DATASET_VERSION:
        raise ValueError("Embedding dataset version mismatch.")

    if report.get("model_name") != EXPECTED_MODEL:
        raise ValueError("Embedding model mismatch.")

    if len(records) != EXPECTED_RECORD_COUNT:
        raise ValueError("Unexpected JSONL record count.")

    if vectors.shape != (
        EXPECTED_RECORD_COUNT,
        EXPECTED_DIMENSION,
    ):
        raise ValueError(f"Unexpected vector shape: {vectors.shape}")

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate embedding IDs.")

    if len(metadata_records) != EXPECTED_RECORD_COUNT:
        raise ValueError("Unexpected metadata record count.")

    record_by_id = {record["id"]: record for record in records}
    metadata_by_id = {
        item["id"]: item["metadata"] for item in metadata_records
    }

    if set(record_by_id) != set(ids):
        raise ValueError("Embedding IDs do not match JSONL IDs.")

    if set(metadata_by_id) != set(ids):
        raise ValueError("Metadata IDs do not match embedding IDs.")

    for chunk_id in ids:
        if not record_by_id[chunk_id].get("text", "").strip():
            raise ValueError(f"Empty document: {chunk_id}")

        if record_by_id[chunk_id]["metadata"] != metadata_by_id[chunk_id]:
            raise ValueError(f"Metadata mismatch: {chunk_id}")

    if not np.isfinite(vectors).all():
        raise ValueError("Non-finite embedding values.")

    return ids, vectors, record_by_id, metadata_by_id


def serialize_metadata(metadata):
    """Convert list-valued fields to JSON strings for storage."""
    result = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, list):
            result[key] = json.dumps(
                value,
                ensure_ascii=False,
            )
        elif isinstance(value, (str, int, float, bool)):
            result[key] = value
        else:
            raise ValueError(
                f"Unsupported metadata type for {key}: "
                f"{type(value).__name__}"
            )

    return result


def main():
    print("Loading embedding artifacts...")
    ids, vectors, records, metadata = load_inputs()

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    print("Opening persistent ChromaDB database...")
    client = chromadb.PersistentClient(path=str(DATABASE_DIR))

    existing = client.get_collection(
        name=COLLECTION_NAME,
    ) if COLLECTION_NAME in [
        item.name for item in client.list_collections()
    ] else None

    if existing is not None:
        print(
            f"Collection already exists with "
            f"{existing.count()} records."
        )
        print("Refusing to overwrite existing collection.")
        print("Use a new versioned collection or review it manually.")
        return

    collection = client.create_collection(
        name=COLLECTION_NAME,
        configuration={
            "hnsw": {
                "space": "cosine",
            }
        },
        metadata={
            "dataset_version": EXPECTED_DATASET_VERSION,
            "embedding_model": EXPECTED_MODEL,
            "embedding_dimension": EXPECTED_DIMENSION,
        },
        embedding_function=None,
    )

    print("Storing records...")

    collection.add(
        ids=ids,
        embeddings=vectors.tolist(),
        documents=[
            records[chunk_id]["text"] for chunk_id in ids
        ],
        metadatas=[
            serialize_metadata(metadata[chunk_id])
            for chunk_id in ids
        ],
    )

    stored_count = collection.count()

    if stored_count != EXPECTED_RECORD_COUNT:
        raise RuntimeError(
            f"Stored {stored_count} records, "
            f"expected {EXPECTED_RECORD_COUNT}."
        )

    print("Reopening collection to verify persistence...")

    reopened = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )

    verification = reopened.get(
        ids=ids,
        include=["documents", "metadatas"],
    )

    returned_ids = set(verification["ids"])

    if returned_ids != set(ids):
        raise RuntimeError("Persisted IDs do not match.")

    if len(verification["documents"]) != EXPECTED_RECORD_COUNT:
        raise RuntimeError("Document persistence verification failed.")

    print("\nChromaDB storage successful!")
    print("Collection:", reopened.name)
    print("Record count:", reopened.count())
    print("Database:", DATABASE_DIR)
    print("Distance metric: cosine")
    print("Embedding model:", EXPECTED_MODEL)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
