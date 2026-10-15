"""
Advanced RAG (step 1): same as Naive RAG, but uses the AdvancedRetriever
(filters from the question + one chunk per trial).

Run all evaluation questions (from the project root):
    python -m src.rag.advanced_rag

Ask one question:
    python -m src.rag.advanced_rag "Find recruiting Phase 3 clinical trials in lung cancer"
"""

import json
import sys
from pathlib import Path

from src.rag.naive_rag import QUESTIONS_PATH, answer_question, gold_coverage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "eval" / "advanced_rag_results.json"


def run_evaluation(retriever, llm=None, questions_path: Path = QUESTIONS_PATH,
                   results_path: Path = RESULTS_PATH) -> list[dict]:
    with open(questions_path, encoding="utf-8") as file:
        questions = json.load(file)

    results = []
    for item in questions:
        print(f"\n{'=' * 60}\n{item['id']} ({item['category']}): {item['question']}")

        result = answer_question(item["question"], retriever, llm)
        gold_ids = item.get("gold_nct_ids", [])
        result.update({
            "id": item["id"],
            "category": item["category"],
            "filters": retriever.last_filters,
            "gold_in_retrieved": gold_coverage(gold_ids, result["retrieved_nct_ids"]),
            "gold_in_answer": gold_coverage(gold_ids, result["cited_nct_ids"]),
        })
        results.append(result)

        print(f"Filters used: {retriever.last_filters['applied']}"
              f"{'  (relaxed)' if retriever.last_filters['relaxed'] else ''}")
        print(f"Retrieved trials: {', '.join(result['retrieved_nct_ids'])}")
        print(f"\nAnswer:\n{result['answer']}")
        if gold_ids:
            g, a = result["gold_in_retrieved"], result["gold_in_answer"]
            print(f"\nCorrect trials found by search: {g['found']}/{g['total']}"
                  f" | cited in answer: {a['found']}/{a['total']}")
        if result["invented_citations"]:
            print(f"WARNING invented citations: {result['invented_citations']}")
        print(f"Time: search {result['retrieval_seconds']}s,"
              f" answer {result['generation_seconds']}s")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {results_path}")
    return results


def main() -> None:
    from src.retrieval.advanced import AdvancedRetriever

    print("Loading index and model...")
    retriever = AdvancedRetriever.from_disk()

    if len(sys.argv) > 1:
        result = answer_question(" ".join(sys.argv[1:]), retriever)
        print(f"\nFilters used: {retriever.last_filters['applied']}")
        print(f"Retrieved trials: {', '.join(result['retrieved_nct_ids'])}")
        print(f"\nAnswer:\n{result['answer']}")
    else:
        run_evaluation(retriever)


if __name__ == "__main__":
    main()