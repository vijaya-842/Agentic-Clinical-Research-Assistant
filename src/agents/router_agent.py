"""Agentic RAG, step 1: a router that picks the right tool for each question.

- "How many ..." questions  -> count_trials tool (plain Python, counts ALL trials)
- every other question      -> Advanced RAG (search + fact sheets + LLM)
"""
import json
import time
from pathlib import Path
import re


from src.tools.count_trials import answer_count_question, is_count_question, load_trials

QUESTIONS_FILE = Path("eval/questions.json")
OUTPUT_FILE = Path("eval/agentic_rag_results.json")

# Personal treatment advice -> fixed safe refusal, no LLM (a guardrail)
PERSONAL_ADVICE_PATTERN = re.compile(
    r"\b(my|our)\s+(father|mother|mom|dad|wife|husband|son|daughter|child|friend|brother|sister|partner|parent)\b"
    r"|\bshould\s+(i|he|she|we|they)\s+(take|get|start|stop|choose|try|use)\b"
    r"|\bwhat\s+should\s+(i|he|she|we)\b",
    re.IGNORECASE,
)
SAFETY_REFUSAL = (
    "I can't give personal medical advice. Treatment decisions for a specific person depend on details "
    "only their care team knows, so please consult their oncologist or doctor. "
    "I can help you find clinical trials or research papers on a cancer type or drug if that is useful."
)

# "List / find / which ... trials" questions want MANY trials, not just 5
LIST_PATTERN = re.compile(r"^\s*(list|find|show)\b|\ball\b|\bwhich\b.*\btrials\b", re.IGNORECASE)
LIST_TOP_K = 10
# Questions about how drugs work / evidence -> research papers, not trials
LITERATURE_PATTERN = re.compile(
    r"\bmechanism\b|\bhow does\b.+\bwork\b|\b(side|adverse) effects?\b|\bresistance\b|\bevidence\b|\bliterature\b|\bpapers?\b",
    re.IGNORECASE,
)

def route(question):
    """Decide which tool should answer the question."""
    if PERSONAL_ADVICE_PATTERN.search(question):
        return "safety_refusal"
    if is_count_question(question):
        return "count_trials"
    if LITERATURE_PATTERN.search(question):
        return "literature_rag"
    if LIST_PATTERN.search(question):
        return "list_rag"
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

def readable_phase(phase):
    """'PHASE3' -> 'Phase 3', 'EARLY_PHASE1' -> 'Early Phase 1'."""
    return phase.replace("EARLY_PHASE", "Early Phase ").replace("PHASE", "Phase ")


def trial_details(nct_ids, all_trials):
    """Write one line per cited trial from the raw data: ID, title, status, phase."""
    by_id = {t["nct_id"]: t for t in all_trials}
    lines = []
    for nct in nct_ids:
        t = by_id.get(nct)
        if not t:
            continue
        phases = ", ".join(readable_phase(p) for p in t.get("phases") or []) or "N/A"
        lines.append(f"- [{nct}] {t.get('title', '')} | Status: {t.get('status', '')} | Phase: {phases}")
    return "\n".join(lines)

def safety_refusal_result(question):
    """A fixed, safe answer for personal medical-advice questions (no search, no LLM)."""
    return {
        "question": question,
        "answer": SAFETY_REFUSAL,
        "retrieved_chunk_ids": [],
        "retrieved_nct_ids": [],
        "cited_nct_ids": [],
        "valid_citations": [],
        "invented_citations": [],
        "retrieval_seconds": 0.0,
        "generation_seconds": 0.0,
        "filters": {},
    }


def answer_question(question, retriever, trial_facts, all_trials, pubmed_retriever=None, top_k=5):
    """The agent: choose a tool, run it, and record which tool was used."""
    tool = route(question)
    if tool == "safety_refusal":
        result = safety_refusal_result(question)
    elif tool == "count_trials":
        result = answer_with_count_tool(question, all_trials)
    elif tool == "literature_rag":
        from src.rag.literature_rag import answer_literature_question
        result = answer_literature_question(question, pubmed_retriever, top_k=top_k)
    else:
        from src.rag.advanced_rag import answer_question as advanced_answer
        k = LIST_TOP_K if tool == "list_rag" else top_k
        result = advanced_answer(question, retriever, trial_facts, top_k=k)
        if tool == "list_rag" and result["valid_citations"]:
            details = trial_details(result["valid_citations"], all_trials)
            result["answer"] += "\n\nTrial details (from ClinicalTrials.gov data):\n" + details
    result["tool"] = tool
    return result


def load_questions(path=QUESTIONS_FILE):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"] if isinstance(data, dict) else data


def run_evaluation(output_path=OUTPUT_FILE, top_k=5, rerank=True, questions_path=QUESTIONS_FILE):
    from src.rag.trial_facts import load_trial_facts
    from src.reranking.cross_encoder import RerankingRetriever
    from src.retrieval.advanced import AdvancedRetriever
    from src.retrieval.pubmed_search import PubMedRetriever

    print("Loading index, model and trial data...")
    retriever = AdvancedRetriever.from_disk()
    trial_facts = load_trial_facts()
    all_trials = load_trials()
    pubmed_retriever = PubMedRetriever.from_disk()

    if rerank:  # FAISS finds 20 candidates, the reranker keeps the best ones
        reranker = RerankingRetriever.load_model()
        retriever = RerankingRetriever(retriever, reranker)
        pubmed_retriever = RerankingRetriever(pubmed_retriever, reranker)
        print("Reranker: ON")

    results = []
    for q in load_questions(questions_path):
        print(f"\n{q['id']}: {q['question']}")
        result = answer_question(q["question"], retriever, trial_facts, all_trials,
                                 pubmed_retriever=pubmed_retriever, top_k=top_k)
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(QUESTIONS_FILE))
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    args = parser.parse_args()
    run_evaluation(output_path=Path(args.output), questions_path=Path(args.questions))