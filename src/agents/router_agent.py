"""Agentic RAG, step 1: a router that picks the right tool for each question.

- "How many ..." questions  -> count_trials tool (plain Python, counts ALL trials)
- every other question      -> Advanced RAG (search + fact sheets + LLM)
"""
import json
import time
from pathlib import Path
import re


from src.tools.count_trials import answer_count_question, is_count_question, load_trials
from src.tools.trial_lookup import answer_trial_lookup, find_nct_ids
from src.tools.small_talk import small_talk_reply, small_talk_result
from src.agents.verifier import verify_answer
from src.agents.multi_hop import is_multi_hop


QUESTIONS_FILE = Path("eval/questions.json")
OUTPUT_FILE = Path("eval/agentic_rag_results.json")

PERSONAL_ADVICE_PATTERN = re.compile(
    r"\b(my|our)\s+(father|mother|mom|dad|wife|husband|son|daughter|child|friend|brother|sister|partner|parent)\b"
    r"|\bshould\s+(i|he|she|we|they)\s+(take|get|start|stop|choose|try|use|join|skip|change|switch|continue"
    r"|increase|decrease|reduce|raise|lower)\b"
    r"|\bwhat\s+should\s+(i|he|she|we)\b"
    r"|\b(for|like)\s+me\b"
    r"|\bmy\s+(cancer|tumou?r|diagnosis|condition|treatment|chemo\w*)\b",
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
    if small_talk_reply(question):
        return "small_talk"
    if PERSONAL_ADVICE_PATTERN.search(question):
        return "safety_refusal"
    if find_nct_ids(question):
        return "trial_lookup"
    if is_count_question(question):
        return "count_trials"
    if is_multi_hop(question):
        return "multi_hop"
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

GENERIC_WORDS = {
    "trial", "trials", "clinical", "recruiting", "completed", "phase", "cancer", "testing", "which",
    "patients", "study", "studies", "include", "intervention", "treatment", "therapy", "about",
}


def drug_matches(question, nct_ids, all_trials):
    """Retrieved trials whose interventions contain a word from the question (e.g. the drug name).
    Checked by code against the raw data, so these are safe to show even if the LLM left them out."""
    words = set(re.findall(r"[a-z][a-z0-9-]{4,}", question.lower())) - GENERIC_WORDS
    by_id = {t["nct_id"]: t for t in all_trials}
    found = []
    for nct in nct_ids:
        interventions = " ".join(str(i) for i in (by_id.get(nct) or {}).get("interventions") or []).lower()
        if words and any(w in interventions for w in words):
            found.append(nct)
    return found


def list_with_details(result, all_trials, question=""):
    """Add code-written status/phase lines for every valid cited trial, plus retrieved trials
    that the LLM left out but whose interventions contain the drug named in the question."""
    cited = result["valid_citations"]
    extra = [n for n in drug_matches(question, result.get("retrieved_nct_ids", []), all_trials) if n not in cited]
    if cited or extra:
        result["answer"] += "\n\nTrial details (from ClinicalTrials.gov data):"
    if cited:
        result["answer"] += "\n" + trial_details(cited, all_trials)
    if extra:
        result["answer"] += ("\nAlso found by search (the drug is listed in the trial's interventions):\n"
                             + trial_details(extra, all_trials))
        result["cited_nct_ids"] = list(dict.fromkeys(result.get("cited_nct_ids", []) + extra))
        result["added_by_code"] = extra
    return result

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
    if tool == "small_talk":
        result = small_talk_result(question)
    elif tool == "safety_refusal":
        result = safety_refusal_result(question)
    elif tool == "trial_lookup":
        result = answer_trial_lookup(question, all_trials)
    elif tool == "count_trials":
        result = answer_with_count_tool(question, all_trials)
    elif tool == "multi_hop":
        from src.agents.multi_hop import answer_multi_hop
        from src.rag.advanced_rag import answer_question as advanced_answer
        from src.rag.literature_rag import answer_literature_question
        result = answer_multi_hop(
            question,
            trial_tool=lambda q: list_with_details(advanced_answer(q, retriever, trial_facts, top_k=LIST_TOP_K), all_trials, q),
            paper_tool=lambda q: answer_literature_question(q, pubmed_retriever, top_k=top_k),
            all_trials=all_trials,
        )
    elif tool == "literature_rag":
        from src.rag.literature_rag import answer_literature_question
        result = answer_literature_question(question, pubmed_retriever, top_k=top_k)
    else:
        from src.rag.advanced_rag import answer_question as advanced_answer
        k = LIST_TOP_K if tool == "list_rag" else top_k
        result = advanced_answer(question, retriever, trial_facts, top_k=k)
        if tool == "list_rag":
            result = list_with_details(result, all_trials, question)
    result["tool"] = tool
    return verify_answer(result, all_trials, tool)


def load_questions(path=QUESTIONS_FILE):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"] if isinstance(data, dict) else data


def run_evaluation(output_path=OUTPUT_FILE, top_k=5, rerank=True, questions_path=QUESTIONS_FILE, hybrid=False):
    from src.rag.trial_facts import load_trial_facts
    from src.retrieval.pipeline import load_retrievers

    print("Loading index, model and trial data...")
    retriever, pubmed_retriever = load_retrievers(hybrid=hybrid, rerank=rerank)
    trial_facts = load_trial_facts()
    all_trials = load_trials()

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
    parser.add_argument("--hybrid", action="store_true", help="add BM25 keyword search (off by default)")
    args = parser.parse_args()
    run_evaluation(output_path=Path(args.output), questions_path=Path(args.questions), hybrid=args.hybrid)