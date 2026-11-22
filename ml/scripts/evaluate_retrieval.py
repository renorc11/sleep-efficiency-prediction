import json
import os

from datetime import datetime, timezone
from pathlib import Path

from search_knowledge import SleepKnowledgeRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_VERSION = os.getenv(
    "KNOWLEDGE_DATASET_VERSION",
    "0.3.1",
)

DATASET_PATH = (
    PROJECT_ROOT
    / f"ml/data/knowledge/sleep_knowledge_dataset_v{DATASET_VERSION}.json"
)

RELEVANCE_PATH = (
    PROJECT_ROOT
    / f"ml/data/knowledge/retrieval_relevance_v{DATASET_VERSION}.json"
)

REPORT_DIR = (
    PROJECT_ROOT
    / f"ml/artifacts/retrieval_evaluation/v{DATASET_VERSION}"
)

COLLECTION_NAME = (
    f"sleep_knowledge_v{DATASET_VERSION.replace('.', '_')}"
)

TOP_K = 3


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_inputs(dataset, relevance, retriever):
    dataset_version = dataset.get("dataset_version")

    if dataset_version != DATASET_VERSION:
        raise ValueError(
            f"Unexpected dataset version: {dataset_version}"
        )

    if relevance.get("dataset_version") != dataset_version:
        raise ValueError(
            "Relevance judgments and dataset versions do not match."
        )

    questions = dataset.get("evaluation_questions", [])
    judgments = relevance.get("relevance_judgments", {})

    question_ids = {
        question["question_id"] for question in questions
    }

    judgment_ids = set(judgments.keys())

    if question_ids != judgment_ids:
        raise ValueError(
            "Evaluation questions and relevance judgments "
            "do not contain matching question IDs."
        )

    dataset_chunk_ids = {
        chunk["chunk_id"]
        for chunk in dataset.get("knowledge_chunks", [])
    }

    collection_chunk_ids = set(
        retriever.collection.get()["ids"]
    )

    for question_id, relevant_ids in judgments.items():

        if not relevant_ids:
            raise ValueError(
                f"No relevant chunks defined for {question_id}."
            )

        unknown_dataset_ids = (
            set(relevant_ids) - dataset_chunk_ids
        )

        unknown_collection_ids = (
            set(relevant_ids) - collection_chunk_ids
        )

        if unknown_dataset_ids:
            raise ValueError(
                f"Unknown dataset chunk IDs for {question_id}: "
                f"{sorted(unknown_dataset_ids)}"
            )

        if unknown_collection_ids:
            raise ValueError(
                f"Relevant chunks missing from ChromaDB "
                f"for {question_id}: "
                f"{sorted(unknown_collection_ids)}"
            )

    return questions, judgments


def calculate_metrics(retrieved_ids, relevant_ids):

    relevant_set = set(relevant_ids)

    retrieved_relevant = [
        chunk_id
        for chunk_id in retrieved_ids
        if chunk_id in relevant_set
    ]

    recall = (
        len(set(retrieved_ids) & relevant_set)
        / len(relevant_set)
    )

    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if chunk_id in relevant_set:
            reciprocal_rank = 1.0 / rank
            break

    hit = 1.0 if retrieved_relevant else 0.0

    return {
        "recall_at_3": recall,
        "mrr_at_3": reciprocal_rank,
        "hit_at_3": hit,
        "retrieved_relevant_ids": retrieved_relevant,
    }


def evaluate():

    print("Loading evaluation dataset...")

    dataset = load_json(DATASET_PATH)
    relevance = load_json(RELEVANCE_PATH)

    print("Initializing retrieval system...")

    retriever = SleepKnowledgeRetriever(version=DATASET_VERSION)

    questions, judgments = validate_inputs(
        dataset,
        relevance,
        retriever,
    )

    print(f"\nEvaluating {len(questions)} questions...")
    print(f"Retrieval depth: Top {TOP_K}")

    results = []

    for question in questions:

        question_id = question["question_id"]
        question_text = question["question"]

        relevant_ids = judgments[question_id]

        retrieved = retriever.search(
            question=question_text,
            top_k=TOP_K,
        )

        retrieved_ids = [
            item["id"] for item in retrieved
        ]

        metrics = calculate_metrics(
            retrieved_ids,
            relevant_ids,
        )

        ranked_results = []

        for rank, item in enumerate(retrieved, start=1):
            ranked_results.append({
                "rank": rank,
                "chunk_id": item["id"],
                "topic": item["topic"],
                "similarity": item["similarity"],
                "is_relevant": (
                    item["id"] in set(relevant_ids)
                ),
            })

        question_result = {
            "question_id": question_id,
            "question": question_text,
            "expected_relevant_ids": relevant_ids,
            "retrieved_results": ranked_results,
            "retrieved_relevant_ids": (
                metrics["retrieved_relevant_ids"]
            ),
            "recall_at_3": metrics["recall_at_3"],
            "mrr_at_3": metrics["mrr_at_3"],
            "hit_at_3": metrics["hit_at_3"],
        }

        results.append(question_result)

        print("\n" + "=" * 65)
        print(f"{question_id}: {question_text}")

        for item in ranked_results:
            status = (
                "RELEVANT"
                if item["is_relevant"]
                else "NOT RELEVANT"
            )

            print(
                f"  Rank {item['rank']}: "
                f"{item['chunk_id']} | "
                f"Similarity: {item['similarity']:.4f} "
                f"| {status}"
            )

        print(
            f"  Recall@3: {metrics['recall_at_3']:.4f}"
        )

        print(
            f"  MRR@3: {metrics['mrr_at_3']:.4f}"
        )

        print(
            f"  Hit@3: {metrics['hit_at_3']:.4f}"
        )

    total_questions = len(results)

    average_recall = (
        sum(item["recall_at_3"] for item in results)
        / total_questions
    )

    average_mrr = (
        sum(item["mrr_at_3"] for item in results)
        / total_questions
    )

    average_hit = (
        sum(item["hit_at_3"] for item in results)
        / total_questions
    )

    report = {
        "evaluation_timestamp": (
            datetime.now(timezone.utc).isoformat()
        ),
        "dataset_version": DATASET_VERSION,
        "judgment_version": relevance["judgment_version"],
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "collection_name": COLLECTION_NAME,
        "total_questions": total_questions,
        "top_k": TOP_K,
        "metrics": {
            "mean_recall_at_3": average_recall,
            "mean_mrr_at_3": average_mrr,
            "mean_hit_at_3": average_hit,
        },
        "question_results": results,
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = REPORT_DIR / "evaluation_report.json"

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 65)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 65)

    print(f"Total questions: {total_questions}")
    print(f"Recall@3: {average_recall:.4f}")
    print(f"MRR@3: {average_mrr:.4f}")
    print(f"Hit@3: {average_hit:.4f}")

    print(f"\nEvaluation report saved to:\n{report_path}")


if __name__ == "__main__":
    evaluate()
