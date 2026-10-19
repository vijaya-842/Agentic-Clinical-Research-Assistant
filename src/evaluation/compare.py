"""
Compare system variants (LLM-only, Naive RAG, ...) on the evaluation questions.

Reads:
    eval/questions.json            questions + gold answers
    eval/<variant>_results.json    one results file per variant

Run from the project root:
    python -m src.evaluation.compare
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "eval"
REPORT_PATH = PROJECT_ROOT / "reports" / "comparison.md"

# Variant name -> results file. Add new variants here as you build them.
VARIANTS = {
    "LLM-only": EVAL_DIR / "llm_only_results.json",
    "Naive RAG": EVAL_DIR / "naive_rag_results.json",
    "Adv. RAG (filters)": EVAL_DIR / "advanced_rag_step1_results.json",
    "Adv. RAG (+ fact sheets)": EVAL_DIR / "advanced_rag_results.json",
    "Agentic (+ count tool)": EVAL_DIR / "agentic_step1_results.json",
    "Agentic (+ list top-10)": EVAL_DIR / "agentic_step2_results.json",
    "Agentic (+ code details)": EVAL_DIR / "agentic_step3_results.json",
    "Agentic (+ PubMed)": EVAL_DIR / "agentic_step4_results.json",
    "LangGraph agent (LLM picks tool)": EVAL_DIR / "langgraph_agent_results.json",
    "Agentic (+ reranker MiniLM)": EVAL_DIR / "agentic_rerank_minilm_results.json",
    "Agentic (+ reranker MedCPT)": EVAL_DIR / "agentic_rerank_medcpt_results.json",
    "Agentic (+ MedCPT + guardrail)": EVAL_DIR / "agentic_rag_results.json",
}

# The harder question set (eval/questions_hard.json): keyword router vs LLM agent
VARIANTS_HARD = {
    "Keyword router": EVAL_DIR / "hard_router_results.json",
    "LangGraph agent": EVAL_DIR / "hard_langgraph_results.json",
}
NCT_PATTERN = re.compile(r"NCT\d{8}")
REFUSAL_WORDS = ["cannot advise", "can't advise", "cannot provide", "oncologist", "doctor"]


def load_json(path: Path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)

def normalize(text: str) -> str:
    """Lowercase, treat '-' as a space, squeeze spaces: 'Third-Generation' -> 'third generation'."""
    return re.sub(r"\s+", " ", text.lower().replace("-", " ")).strip()


def fact_found(fact, answer: str) -> bool:
    """A fact is a phrase, or a list of acceptable phrases (any one is enough)."""
    options = fact if isinstance(fact, list) else [fact]
    answer_norm = normalize(answer)
    for option in options:
        if option.isdigit():
            # a number must stand alone, and digits inside NCT IDs do not count
            without_ids = NCT_PATTERN.sub(" ", answer)
            if re.search(rf"(?<!\d){option}(?!\d)", without_ids):
                return True
        elif normalize(option) in answer_norm:
            return True
    return False

def score_answer(question: dict, result: dict) -> dict:
    """
    Score one answer against the gold information for its question.

    - gold_ids:  how many gold trial IDs the answer mentions
    - facts:     how many key facts (or the exact gold answer) appear in the answer
    - unsupported_ids: trial IDs in the answer that cannot be trusted
        (for RAG: IDs not in the retrieved sources; for LLM-only: IDs not in the gold list)
    - refusal_ok: for out-of-scope questions, does the answer point to a doctor?
    """
    answer = result.get("answer", "")
    answer_lower = answer.lower()
    cited = list(dict.fromkeys(NCT_PATTERN.findall(answer)))

    gold_ids = question.get("gold_nct_ids", [])
    facts = list(question.get("key_facts", []))
    if question.get("gold_type") == "exact" and question.get("gold_answer"):
        facts.append(question["gold_answer"])

    if "invented_citations" in result:
        unsupported = result["invented_citations"]
    else:
        unsupported = [nct_id for nct_id in cited if nct_id not in gold_ids]

    score = {
        "gold_ids_found": sum(1 for nct_id in gold_ids if nct_id in cited),
        "gold_ids_total": len(gold_ids),
        "facts_found": sum(1 for fact in facts if fact_found(fact, answer)),
        "facts_total": len(facts),
        "unsupported_ids": unsupported,
    }
    if question.get("category") == "out_of_scope":
        score["refusal_ok"] = any(word in answer_lower for word in REFUSAL_WORDS)
    return score


def compare(questions: list[dict], variant_results: dict[str, list[dict]]) -> dict:
    """Score every variant on every question."""
    scores: dict[str, dict[str, dict]] = {}
    for variant, results in variant_results.items():
        by_id = {result["id"]: result for result in results}
        scores[variant] = {
            question["id"]: score_answer(question, by_id[question["id"]])
            for question in questions
            if question["id"] in by_id
        }
    return scores


def _cell(found: int, total: int) -> str:
    return f"{found}/{total}" if total else "-"


def build_report(questions: list[dict], scores: dict) -> str:
    """Make a Markdown table: one row per question, columns per variant."""
    variants = list(scores)
    header = ["Q", "Category"]
    for variant in variants:
        header += [f"{variant} trials", f"{variant} facts"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    totals = {variant: {"ids": 0, "ids_total": 0, "facts": 0, "facts_total": 0,
                        "unsupported": 0} for variant in variants}

    for question in questions:
        row = [question["id"], question["category"]]
        for variant in variants:
            s = scores[variant].get(question["id"])
            if s is None:
                row += ["n/a", "n/a"]
                continue
            facts = _cell(s["facts_found"], s["facts_total"])
            if "refusal_ok" in s:
                facts = "refused OK" if s["refusal_ok"] else "no refusal"
            row += [_cell(s["gold_ids_found"], s["gold_ids_total"]), facts]

            t = totals[variant]
            t["ids"] += s["gold_ids_found"]
            t["ids_total"] += s["gold_ids_total"]
            t["facts"] += s["facts_found"]
            t["facts_total"] += s["facts_total"]
            t["unsupported"] += len(s["unsupported_ids"])
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "| Variant | Gold trials cited | Key facts found | Unsupported trial IDs |",
              "|---|---|---|---|"]
    for variant in variants:
        t = totals[variant]
        lines.append(
            f"| {variant} | {t['ids']}/{t['ids_total']} | "
            f"{t['facts']}/{t['facts_total']} | {t['unsupported']} |"
        )
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hard", action="store_true", help="score the harder question set")
    args = parser.parse_args()

    if args.hard:
        questions = load_json(EVAL_DIR / "questions_hard.json")
        variants, report_path = VARIANTS_HARD, REPORT_PATH.with_name("comparison_hard.md")
    else:
        questions = load_json(EVAL_DIR / "questions.json")
        variants, report_path = VARIANTS, REPORT_PATH

    variant_results = {}
    for variant, path in variants.items():
        if path.exists():
            variant_results[variant] = load_json(path)
        else:
            print(f"Skipping {variant}: {path.name} not found yet")

    scores = compare(questions, variant_results)
    report = build_report(questions, scores)

    print(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("# Variant comparison\n\n" + report + "\n", encoding="utf-8")
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()