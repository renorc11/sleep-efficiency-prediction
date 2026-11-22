
import os
import argparse
from pathlib import Path

import chromadb
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_DIR = PROJECT_ROOT / "ml/artifacts/chroma_db"

DATASET_VERSION = os.getenv("KNOWLEDGE_DATASET_VERSION", "0.3.1")

COLLECTION_NAME = (
    f"sleep_knowledge_v{DATASET_VERSION.replace('.', '_')}"
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"

QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)

EXPECTED_DIMENSION = 384
EXPECTED_RECORD_COUNT = 17
DEFAULT_TOP_K = 3


class SleepKnowledgeRetriever:

    def __init__(self, version: str = DATASET_VERSION):
        collection_name = (f"sleep_knowledge_v{version.replace('.', '_')}")
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        print(f"Loading embedding model on {device}...")

        self.model = SentenceTransformer(
            MODEL_NAME,
            device=device,
        )

        self.client = chromadb.PersistentClient(
            path=str(DATABASE_DIR)
        )

        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=None,
        )
        
        if self.collection.count() != EXPECTED_RECORD_COUNT:
            raise RuntimeError(
                "Unexpected collection record count."
            )

        print(
            f"Connected to collection: "
            f"{self.collection.name}"
        )

    def search(self, question: str, top_k: int = DEFAULT_TOP_K):
        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        if not 1 <= top_k <= 17:
            raise ValueError("top_k must be between 1 and 17.")

        query_text = QUERY_INSTRUCTION + question

        query_vector = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        if query_vector.shape != (1, EXPECTED_DIMENSION):
            raise RuntimeError(
                f"Unexpected query shape: {query_vector.shape}"
            )

        results = self.collection.query(
            query_embeddings=query_vector.tolist(),
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        retrieved = []

        for chunk_id, document, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - float(distance)

            retrieved.append({
                "id": chunk_id,
                "topic": metadata.get("topic", ""),
                "category": metadata.get("category", ""),
                "similarity": similarity,
                "distance": float(distance),
                "text": document,
                "metadata": metadata,
            })

        return retrieved


def main():
    parser = argparse.ArgumentParser(
        description="Search the sleep knowledge database."
    )

    parser.add_argument(
        "--question",
        required=True,
        help="Natural-language sleep question.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of results to retrieve.",
    )
    parser.add_argument(
    "--version",
    default=DATASET_VERSION,
    help="Knowledge dataset version (e.g. 0.3.1).",)
   
    
    args = parser.parse_args()

    retriever = SleepKnowledgeRetriever(version=args.version)

    results = retriever.search(
        question=args.question,
        top_k=args.top_k,
    )

    print("\nRETRIEVAL RESULTS")
    print("=" * 65)

    for index, result in enumerate(results, start=1):
        print(f"\nRank: {index}")
        print(f"Chunk ID: {result['id']}")
        print(f"Topic: {result['topic']}")
        print(f"Category: {result['category']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Distance: {result['distance']:.4f}")
        print("Full content:")
        print(result["text"])
        print("-" * 65)
        print("\nMetadata:")
        for key, value in result["metadata"].items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
