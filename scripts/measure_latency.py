"""Measure speed: model loading time + time per question, grouped by tool.

Run:     python scripts/measure_latency.py
Output:  reports/latency.md
"""
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.router_agent import QUESTIONS_FILE, answer_question, load_questions  # noqa: E402
from src.tools.count_trials import load_trials  # noqa: E402

REPORT = Path("reports/latency.md")
EXTRA_QUESTIONS = ["hi", "Tell me about NCT05086614"]  # small talk + trial lookup


def memory_mb():
    """RAM used by this Python process (needs psutil; returns None if it is missing)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 2
    except ImportError:
        return None


def main():
    from src.rag.trial_facts import load_trial_facts
    from src.retrieval.pipeline import load_retrievers

    print("Loading models and data...")
    start = time.perf_counter()
    retriever, papers = load_retrievers(hybrid=False, rerank=True)
    trial_facts = load_trial_facts()
    all_trials = load_trials()
    load_seconds = time.perf_counter() - start
    ram = memory_mb()

    questions = [q["question"] for q in load_questions(QUESTIONS_FILE)] + EXTRA_QUESTIONS
    rows = []
    for i, question in enumerate(questions, 1):
        t = time.perf_counter()
        result = answer_question(question, retriever, trial_facts, all_trials, pubmed_retriever=papers)
        seconds = time.perf_counter() - t
        rows.append((question, result["tool"], seconds))
        print(f"[{i}/{len(questions)}] {seconds:6.1f}s  {result['tool']:<16} {question[:60]}")

    by_tool = defaultdict(list)
    for _, tool, seconds in rows:
        by_tool[tool].append(seconds)

    lines = [
        "# Latency report",
        "",
        "Setup: keyword router, dense search + MedCPT reranker, qwen2.5:3b via Ollama, CPU.",
        "The first AI question also includes loading the LLM into memory (warm-up).",
        "",
        f"- Model + data loading (once at start): **{load_seconds:.1f} s**",
    ]
    if ram is not None:
        lines.append(f"- RAM used by the app after loading: **{ram:,.0f} MB**")
    lines += [
        f"- Total for {len(rows)} questions: **{sum(s for *_, s in rows):.1f} s**",
        "",
        "## By tool",
        "",
        "| Tool | Questions | Average (s) | Slowest (s) |",
        "|---|---|---|---|",
    ]
    for tool, times in sorted(by_tool.items(), key=lambda kv: statistics.mean(kv[1])):
        lines.append(f"| {tool} | {len(times)} | {statistics.mean(times):.2f} | {max(times):.2f} |")
    lines += ["", "## Per question", "", "| # | Tool | Seconds | Question |", "|---|---|---|---|"]
    for i, (question, tool, seconds) in enumerate(rows, 1):
        lines.append(f"| {i} | {tool} | {seconds:.2f} | {question} |")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[:lines.index("## Per question")]))
    print(f"\nSaved report to {REPORT}")


if __name__ == "__main__":
    main()