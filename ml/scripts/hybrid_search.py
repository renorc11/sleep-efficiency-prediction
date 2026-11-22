
import argparse
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from search_knowledge import (
    SleepKnowledgeRetriever,
    PROJECT_ROOT,
)


DEFAULT_TOP_K = 3
RRF_K = 60


def tokenize(text: str):
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


class HybridSleepRetriever:

    def __init__(self):

        print("Initializing hybrid retrieval system...")

        self.semantic_retriever = SleepKnowledgeRetriever()

        stored_data = self.semantic_retriever.collection.get(
            include=["documents", "metadatas"]
        )

        self.documents = stored_data["documents"]
        self.metadatas = stored_data["metadatas"]
        self.chunk_ids = stored_data["ids"]

        if len(self.chunk_ids) != 17:
            raise RuntimeError(
                "Unexpected number of documents in collection."
            )

        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise RuntimeError("Duplicate chunk IDs detected.")

        tokenized_documents = [
            tokenize(document)
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

        self.document_lookup = {
            chunk_id: {
                "id": chunk_id,
                "text": document,
                "metadata": metadata,
                "topic": metadata.get("topic", ""),
                "category": metadata.get("category", ""),
            }
            for chunk_id, document, metadata in zip(
                self.chunk_ids,
                self.documents,
                self.metadatas,
            )
        }

        print(
            f"BM25 index created with "
            f"{len(self.chunk_ids)} documents."
        )

    def search(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        candidate_k: int = 17,
    ):

        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        if not 1 <= top_k <= 17:
            raise ValueError("top_k must be between 1 and 17.")

        if not top_k <= candidate_k <= 17:
            raise ValueError(
                "candidate_k must be between top_k and 17."
            )

        query_tokens = tokenize(question)

        if not query_tokens:
            raise ValueError(
                "Question contains no searchable tokens."
            )

        # 1. BM25 retrieval
        bm25_scores = self.bm25.get_scores(query_tokens)

        bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda index: bm25_scores[index],
            reverse=True,
        )[:candidate_k]

        bm25_ranks = {
            self.chunk_ids[index]: rank
            for rank, index in enumerate(
                bm25_indices,
                start=1,
            )
        }

        # 2. Semantic retrieval
        semantic_results = self.semantic_retriever.search(
            question=question,
            top_k=candidate_k,
        )

        semantic_ranks = {
            item["id"]: rank
            for rank, item in enumerate(
                semantic_results,
                start=1,
            )
        }

        semantic_lookup = {
            item["id"]: item
            for item in semantic_results
        }

        # 3. Reciprocal Rank Fusion
        fused_scores = {}

        for chunk_id in self.chunk_ids:

            score = 0.0

            if chunk_id in bm25_ranks:
                score += 1.0 / (
                    RRF_K + bm25_ranks[chunk_id]
                )

            if chunk_id in semantic_ranks:
                score += 1.0 / (
                    RRF_K + semantic_ranks[chunk_id]
                )

            fused_scores[chunk_id] = score

        ranked_ids = sorted(
            fused_scores,
            key=lambda chunk_id: fused_scores[chunk_id],
            reverse=True,
        )[:top_k]

        # 4. Format results
        results = []

        for chunk_id in ranked_ids:

            document = self.document_lookup[chunk_id]
            semantic_result = semantic_lookup.get(chunk_id)

            results.append({
                **document,
                "hybrid_score": fused_scores[chunk_id],
                "bm25_rank": bm25_ranks.get(chunk_id),
                "semantic_rank": semantic_ranks.get(chunk_id),
                "semantic_similarity": (
                    semantic_result["similarity"]
                    if semantic_result
                    else None
                ),
            })

        return results


def main():

    parser = argparse.ArgumentParser(
        description="Hybrid BM25 and semantic sleep knowledge search."
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
    )

    args = parser.parse_args()

    retriever = HybridSleepRetriever()

    results = retriever.search(
        question=args.question,
        top_k=args.top_k,
    )

    print("\nHYBRID RETRIEVAL RESULTS")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):

        print(f"\nRank: {rank}")
        print(f"Chunk ID: {result['id']}")
        print(f"Topic: {result['topic']}")
        print(
            f"Hybrid score: {result['hybrid_score']:.6f}"
        )
        print(f"BM25 rank: {result['bm25_rank']}")
        print(f"Semantic rank: {result['semantic_rank']}")
        print(
            f"Semantic similarity: "
            f"{result['semantic_similarity']}"
        )
        print("\nContent:")
        print(result["text"])
        print("-" * 70)


if __name__ == "__main__":
    main()
