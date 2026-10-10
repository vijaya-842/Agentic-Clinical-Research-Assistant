"""
Naive RAG: search the FAISS index for the top chunks, then ask the local LLM
to answer ONLY from those chunks, citing trial IDs.

Run all evaluation questions (from the project root):
    python -m src.rag.naive_rag

Ask one question:
    python -m src.rag.naive_rag "Which trials test osimertinib?"
"""

import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.json"
RESULTS_PATH = PROJECT_ROOT / "eval" / "naive_rag_results.json"

NCT_PATTERN = re.compile(r"NCT\d{8}")

SYSTEM_PROMPT = """You are a biomedical research assistant for researchers.

Rules:
1. Answer ONLY using the numbered sources provided. Do not use outside knowledge.
2. After every fact, cite the trial ID in square brackets, for example [NCT01234567].
3. Never invent trial IDs. Only cite IDs that appear in the sources.
4. If the sources do not contain the answer, say exactly:
   "The retrieved sources do not contain enough information to answer this question."
5. If the question asks for treatment advice for a specific person, do not recommend
   any treatment. Say you cannot advise on an individual's treatment and that they
   should talk to their oncologist.
"""


def extract_nct_ids(text: str) -> list[str]:
    """Find every NCT ID in a text, in order, without duplicates."""
    seen = []
    for nct_id in NCT_PATTERN.findall(text):
        if nct_id not in seen:
            seen.append(nct_id)
    return seen


def build_prompt(question: str, results: list) -> str:
    """Number each retrieved chunk so the LLM can cite it."""
    sources = []
    for number, result in enumerate(results, start=1):
        nct_id = result.metadata.get("nct_id", "")
        section = result.metadata.get("section", "")
        sources.append(f"[{number}] ({nct_id}, {section})\n{result.text}")

    return (
        "SOURCES:\n\n"
        + "\n\n".join(sources)
        + f"\n\nQUESTION: {question}\n\n"
        + "Answer using only the sources above and cite trial IDs like [NCT01234567]."
    )


def check_citations(answer: str, results: list) -> dict:
    """
    Compare the trial IDs cited in the answer with the trial IDs we retrieved.
    Any cited ID that was NOT retrieved is a hallucinated citation.
    """
    retrieved = {result.metadata.get("nct_id") for result in results}
    cited = extract_nct_ids(answer)
    return {
        "cited_nct_ids": cited,
        "valid_citations": [nct_id for nct_id in cited if nct_id in retrieved],
        "invented_citations": [nct_id for nct_id in cited if nct_id not in retrieved],
    }


def gold_coverage(gold_ids: list[str], found_ids: list[str]) -> dict:
    """How many of the correct trial IDs appear in a list of found IDs."""
    if not gold_ids:
        return {"found": 0, "total": 0}
    found = [nct_id for nct_id in gold_ids if nct_id in set(found_ids)]
    return {"found": len(found), "total": len(gold_ids)}


def answer_question(question: str, retriever, llm=None, top_k: int = 5) -> dict:
    """Retrieve chunks, ask the LLM, and check its citations."""
    if llm is None:
        # Imported here so tests can run without Ollama installed.
        from src.llm.ollama_client import generate_response as llm

    start = time.time()
    results = retriever.search(question, top_k=top_k)
    retrieval_seconds = time.time() - start

    prompt = build_prompt(question, results)

    start = time.time()
    answer = llm(prompt=prompt, system_prompt=SYSTEM_PROMPT)
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
        "retrieval_seconds": round(retrieval_seconds, 3),
        "generation_seconds": round(generation_seconds, 1),
    }


def run_evaluation(retriever, llm=None, questions_path: Path = QUESTIONS_PATH,
                   results_path: Path = RESULTS_PATH) -> list[dict]:
    """Answer every evaluation question and save the results."""
    with open(questions_path, encoding="utf-8") as file:
        questions = json.load(file)

    results = []
    for item in questions:
        print(f"\n{'=' * 60}\n{item['id']} ({item['category']}): {item['question']}")

        result = answer_question(item["question"], retriever, llm)
        gold_ids = item.get("gold_nct_ids", [])
        result.update(
            {
                "id": item["id"],
                "category": item["category"],
                "gold_in_retrieved": gold_coverage(gold_ids, result["retrieved_nct_ids"]),
                "gold_in_answer": gold_coverage(gold_ids, result["cited_nct_ids"]),
            }
        )
        results.append(result)

        print(f"\nRetrieved trials: {', '.join(result['retrieved_nct_ids'])}")
        print(f"\nAnswer:\n{result['answer']}")
        if gold_ids:
            g = result["gold_in_retrieved"]
            a = result["gold_in_answer"]
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
    from src.retrieval.dense import DenseRetriever

    print("Loading index and model...")
    retriever = DenseRetriever.from_disk()

    if len(sys.argv) > 1:
        result = answer_question(" ".join(sys.argv[1:]), retriever)
        print(f"\nRetrieved trials: {', '.join(result['retrieved_nct_ids'])}")
        print(f"\nAnswer:\n{result['answer']}")
        if result["invented_citations"]:
            print(f"\nWARNING invented citations: {result['invented_citations']}")
    else:
        run_evaluation(retriever)


if __name__ == "__main__":
    main()