"""Agentic RAG, step 1: a router that picks the right tool for each question.

- "How many ..." questions  -> count_trials tool (plain Python, counts ALL trials)
- every other question      -> Advanced RAG (search + fact sheets + LLM)
"""
import json
import time
from pathlib import Path

from src.tools.count_trials import answer_count_question, is_count_question, load_trials

QUESTIONS_FILE = Path("eval/questions.json")
OUTPUT_FILE = Path("eval/agentic_rag_results.json")


def route(question):
    """Decide which tool should answer the question."""
    if is_count_question(question):
        return "count_trials"
    return "advanced_rag"


def coverage(gold_ids, found_ids):
    """How many gold trials appear in a list, e.g. {'found': 2, 'total': 4}."""
    found = [g for g in gold_ids if g in found_ids]
    return {"found": len(found), "total": len(gold_ids)}


def answer_with_count_tool(question, all_trials):
    """Use the counting tool and return a result in the same shape as Advanced RAG."""
    start = time.time()
    result = answer_count_question(question, all_trials)
    ids = result["example_ids"]
    return {
        "question": question,
        "answer": result["answer"],
        "retrieved_chunk_ids": [],
        "retrieved_nct_ids": ids,
        "cited_nct_ids": ids,
        "valid_citations": ids,
        "invented_citations": [],
        "retrieval_seconds": round(time.time() - start, 2),
        "generation_seconds": 0.0,
        "filters": result["filters"],
    }


def answer_question(question, retriever, trial_facts, all_trials, top_k=5):
    """The agent: choose a tool, run it, and record which tool was used."""
    tool = route(question)
    if tool == "count_trials":
        result = answer_with_count_tool(question, all_trials)
    else:
        from src.rag.advanced_rag import answer_question as advanced_answer
        result = advanced_answer(question, retriever, trial_facts, top_k=top_k)
    result["tool"] = tool
    return result


def load_questions(path=QUESTIONS_FILE):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"] if isinstance(data, dict) else data


def run_evaluation(output_path=OUTPUT_FILE, top_k=5):
    from src.rag.trial_facts import load_trial_facts
    from src.retrieval.advanced import AdvancedRetriever

    print("Loading index, model and trial data...")
    retriever = AdvancedRetriever.from_disk()
    trial_facts = load_trial_facts()
    all_trials = load_trials()

    results = []
    for q in load_questions():
        print(f"\n{q['id']}: {q['question']}")
        result = answer_question(q["question"], retriever, trial_facts, all_trials, top_k=top_k)
        gold = q.get("gold_nct_ids", [])
        result["id"] = q["id"]
        result["category"] = q.get("category")
        result["gold_in_retrieved"] = coverage(gold, result["retrieved_nct_ids"])
        result["gold_in_answer"] = coverage(gold, result["cited_nct_ids"])
        print(f"Tool: {result['tool']}")
        print(f"Answer: {result['answer']}")
        results.append(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} results to {output_path}")


if __name__ == "__main__":
    run_evaluation()