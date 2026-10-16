"""
Advanced RAG:
  - AdvancedRetriever: filters from the question + one chunk per trial
  - Trial fact sheets: sponsor, status, phase and primary outcomes for every
    retrieved trial, so the LLM never has to guess or borrow these details

Run all evaluation questions (from the project root):
    python -m src.rag.advanced_rag

Ask one question:
    python -m src.rag.advanced_rag "Find recruiting Phase 3 clinical trials in lung cancer"
"""

import json
import sys
import time
from pathlib import Path

from src.rag.naive_rag import QUESTIONS_PATH, SYSTEM_PROMPT, check_citations, gold_coverage
from src.rag.trial_facts import build_prompt_with_facts, load_trial_facts

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "eval" / "advanced_rag_results.json"

ADVANCED_SYSTEM_PROMPT = SYSTEM_PROMPT + """6. Each source starts with a FACT SHEET. Use the fact sheet for a trial's
   sponsor, status, phase and primary outcomes.
7. Describe each trial ONLY with information from its own source.
   Never take a detail from one trial and attach it to another trial.
"""


def answer_question(question: str, retriever, trial_facts: dict, llm=None, top_k: int = 5) -> dict:
    """Retrieve chunks, add fact sheets, ask the LLM, and check its citations."""
    if llm is None:
        from src.llm.ollama_client import generate_response as llm

    start = time.time()
    results = retriever.search(question, top_k=top_k)
    retrieval_seconds = time.time() - start

    prompt = build_prompt_with_facts(question, results, trial_facts)

    start = time.time()
    answer = llm(prompt=prompt, system_prompt=ADVANCED_SYSTEM_PROMPT)
    generation_seconds = time.time() - start

    retrieved_nct_ids = []
    for result in results:
        nct_id = result.metadata.get("nct_id")
        if nct_id and nct_id not in retrieved_nct_ids:
            retrieved_nct_ids.append(nct_id)

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunk_ids": [result.chunk_id for result in results],
        "retrieved_nct_ids": retrieved_nct_ids,
        **check_citations(answer, results),
        "filters": getattr(retriever, "last_filters", {}),
        "retrieval_seconds": round(retrieval_seconds, 3),
        "generation_seconds": round(generation_seconds, 1),
    }


def run_evaluation(retriever, trial_facts: dict, llm=None, questions_path: Path = QUESTIONS_PATH,
                   results_path: Path = RESULTS_PATH) -> list[dict]:
    with open(questions_path, encoding="utf-8") as file:
        questions = json.load(file)

    results = []
    for item in questions:
        print(f"\n{'=' * 60}\n{item['id']} ({item['category']}): {item['question']}")

        result = answer_question(item["question"], retriever, trial_facts, llm)
        gold_ids = item.get("gold_nct_ids", [])
        result.update({
            "id": item["id"],
            "category": item["category"],
            "gold_in_retrieved": gold_coverage(gold_ids, result["retrieved_nct_ids"]),
            "gold_in_answer": gold_coverage(gold_ids, result["cited_nct_ids"]),
        })
        results.append(result)

        filters = result["filters"]
        if filters:
            print(f"Filters used: {filters.get('applied')}"
                  f"{'  (relaxed)' if filters.get('relaxed') else ''}")
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

    print("Loading index, model and trial facts...")
    retriever = AdvancedRetriever.from_disk()
    trial_facts = load_trial_facts()

    if len(sys.argv) > 1:
        result = answer_question(" ".join(sys.argv[1:]), retriever, trial_facts)
        print(f"\nFilters used: {result['filters'].get('applied')}")
        print(f"Retrieved trials: {', '.join(result['retrieved_nct_ids'])}")
        print(f"\nAnswer:\n{result['answer']}")
    else:
        run_evaluation(retriever, trial_facts)


if __name__ == "__main__":
    main()