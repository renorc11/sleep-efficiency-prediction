import os
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_VERSION = os.getenv("KNOWLEDGE_DATASET_VERSION", "0.3.1")

DATASET_PATH = (
    PROJECT_ROOT
    / f"ml/data/knowledge/sleep_knowledge_v{DATASET_VERSION}.jsonl"
)

MASTER_PATH = (
    PROJECT_ROOT
    / f"ml/data/knowledge/sleep_knowledge_dataset_v{DATASET_VERSION}.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / f"ml/artifacts/knowledge_embeddings/v{DATASET_VERSION}"
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EXPECTED_DIMENSION = 384
EXPECTED_RECORDS = 17


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    if not MASTER_PATH.exists():
        raise FileNotFoundError(MASTER_PATH)

    with MASTER_PATH.open(encoding="utf-8") as file:
        master = json.load(file)

    if master.get("dataset_version") != DATASET_VERSION:
        raise ValueError("Unexpected master dataset version.")

    with DATASET_PATH.open(encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if len(records) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS} records, "
            f"found {len(records)}."
        )

    master_ids = {item["chunk_id"] for item in master["knowledge_chunks"]}

    record_ids = [record.get("id") for record in records]

    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Duplicate chunk IDs found.")

    if set(record_ids) != master_ids:
        raise ValueError(
            "JSONL chunk IDs do not match master dataset."
        )

    for record in records:
        if not isinstance(record.get("text"), str):
            raise ValueError(
                f"Invalid text in {record.get('id')}"
            )

        if not record["text"].strip():
            raise ValueError(
                f"Empty text in {record.get('id')}"
            )

        if not isinstance(record.get("metadata"), dict):
            raise ValueError(
                f"Invalid metadata in {record.get('id')}"
            )

    return records


def main():
    print("Loading and validating knowledge dataset...")

    records = load_dataset()

    ids = [record["id"] for record in records]
    texts = [record["text"] for record in records]

    print(f"Validated records: {len(records)}")

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    embeddings = model.encode(
        texts,
        batch_size=8,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.shape != (
        EXPECTED_RECORDS,
        EXPECTED_DIMENSION,
    ):
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Non-finite embedding values found.")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Embeddings are not normalized.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    embedding_path = OUTPUT_DIR / "embeddings.npz"
    metadata_path = OUTPUT_DIR / "metadata.json"
    report_path = OUTPUT_DIR / "generation_report.json"

    # Save embeddings in compressed NumPy format.
    np.savez_compressed(
        embedding_path,
        ids=np.asarray(ids),
        embeddings=embeddings,
    )

    # Preserve original chunk metadata.
    metadata_records = [
        {
            "id": record["id"],
            "metadata": record["metadata"],
        }
        for record in records
    ]

    metadata_path.write_text(
        json.dumps(metadata_records, indent=2),
        encoding="utf-8",
    )

    report = {
        "dataset_version": DATASET_VERSION,
        "model_name": MODEL_NAME,
        "device": device,
        "embedding_dimension": EXPECTED_DIMENSION,
        "record_count": len(records),
        "normalized": True,
        "embedding_shape": list(embeddings.shape),
        "all_embeddings_finite": True,
        "all_norms_approximately_one": True,
        "files": [
            "embeddings.npz",
            "metadata.json",
        ],
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nEmbedding generation successful!")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Saved: {embedding_path.name}")
    print(f"Saved: {metadata_path.name}")
    print(f"Saved: {report_path.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
