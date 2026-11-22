
import json
from datetime import datetime, timezone
from pathlib import Path

from hybrid_search import HybridSleepRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "ml/data/knowledge/sleep_knowledge_dataset_v0.3.0.json"
)

RELEVANCE_PATH = (
    PROJECT_ROOT
    / "ml/data/knowledge/retrieval_relevance_v0.3.0.json"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "ml/artifacts/retrieval_evaluation/v0.3.0"
)

TOP_K = 3
CANDIDATE_K = 17


def load_json(path):

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def calculate_metrics(retrieved_ids, relevant_ids):

    relevant_set = set(relevant_ids)

    relevant_retrieved = (
        set(retrieved_ids) & relevant_set
    )

    recall = (
        len(relevant_retrieved) / len(relevant_set)
    )

    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if chunk_id in relevant_set:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "recall_at_3": recall,
        "mrr_at_3": reciprocal_rank,
        "hit_at_3": (
            1.0 if relevant_retrieved else 0.0
        ),
    }


def evaluate():

    dataset = load_json(DATASET_PATH)
    relevance = load_json(RELEVANCE_PATH)

    if dataset.get("dataset_version") != "0.3.0":
        raise ValueError("Unexpected dataset version.")

    if relevance.get("dataset_version") != "0.3.0":
        raise ValueError("Relevance judgment version mismatch.")

    questions = dataset["evaluation_questions"]
    judgments = relevance["relevance_judgments"]

    question_ids = {
        question["question_id"]
        for question in questions
    }

    if question_ids != set(judgments):
        raise ValueError(
            "Evaluation questions and judgments do not match."
        )

    retriever = HybridSleepRetriever()

    collection_ids = set(
        retriever.semantic_retriever.collection.get()["ids"]
    )

    dataset_ids = {
        chunk["chunk_id"]
        for chunk in dataset["knowledge_chunks"]
    }

    for question_id, relevant_ids in judgments.items():

        if not relevant_ids:
            raise ValueError(
                f"No relevant chunks for {question_id}."
            )

        if not set(relevant_ids).issubset(dataset_ids):
            raise ValueError(
                f"Unknown dataset chunk ID in {question_id}."
            )

        if not set(relevant_ids).issubset(collection_ids):
            raise ValueError(
                f"Relevant chunk missing from ChromaDB: {question_id}"
            )

    results = []

    print("\nHYBRID RETRIEVAL EVALUATION")
    print("=" * 70)

    for question in questions:

        question_id = question["question_id"]
        question_text = question["question"]

        relevant_ids = judgments[question_id]

        retrieved = retriever.search(
            question=question_text,
            top_k=TOP_K,
            candidate_k=CANDIDATE_K,
        )

        retrieved_ids = [
            item["id"] for item in retrieved
        ]

        metrics = calculate_metrics(
            retrieved_ids,
            relevant_ids,
        )

        question_result = {
            "question_id": question_id,
            "question": question_text,
            "expected_relevant_ids": relevant_ids,
            "retrieved_ids": retrieved_ids,
            "retrieved_results": [
                {
                    "rank": rank,
                    "chunk_id": item["id"],
                    "hybrid_score": item["hybrid_score"],
                    "bm25_rank": item["bm25_rank"],
                    "semantic_rank": item["semantic_rank"],
                }
                for rank, item in enumerate(
                    retrieved,
                    start=1,
                )
            ],
            **metrics,
        }

        results.append(question_result)

        print(f"\n{question_id}: {question_text}")

        for item in question_result["retrieved_results"]:

            status = (
                "RELEVANT"
                if item["chunk_id"] in relevant_ids
                else "NOT RELEVANT"
            )

            print(
                f"  Rank {item['rank']}: "
                f"{item['chunk_id']} | {status}"
            )

        print(f"  Recall@3: {metrics['recall_at_3']:.4f}")
        print(f"  MRR@3: {metrics['mrr_at_3']:.4f}")
        print(f"  Hit@3: {metrics['hit_at_3']:.4f}")

    total = len(results)

    mean_recall = sum(
        item["recall_at_3"] for item in results
    ) / total

    mean_mrr = sum(
        item["mrr_at_3"] for item in results
    ) / total

    mean_hit = sum(
        item["hit_at_3"] for item in results
    ) / total

    report = {
        "evaluation_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset_version": "0.3.0",
        "judgment_version": relevance["judgment_version"],
        "retrieval_method": "BM25 + Semantic + RRF",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "collection_name": "sleep_knowledge_v0_3",
        "total_questions": total,
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "rrf_constant": 60,
        "metrics": {
            "mean_recall_at_3": mean_recall,
            "mean_mrr_at_3": mean_mrr,
            "mean_hit_at_3": mean_hit,
        },
        "question_results": results,
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        REPORT_DIR / "hybrid_evaluation_report.json"
    )

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("HYBRID EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Questions: {total}")
    print(f"Recall@3: {mean_recall:.4f}")
    print(f"MRR@3: {mean_mrr:.4f}")
    print(f"Hit@3: {mean_hit:.4f}")

    print(f"\nReport saved to:\n{report_path}")


if __name__ == "__main__":
    evaluate()
