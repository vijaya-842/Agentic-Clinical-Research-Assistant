import json

from src.rag.naive_rag import (
    answer_question,
    build_prompt,
    check_citations,
    extract_nct_ids,
    gold_coverage,
    run_evaluation,
)
from src.retrieval.dense import SearchResult


def _result(nct_id: str, section: str = "summary", text: str = "some text") -> SearchResult:
    return SearchResult(
        chunk_id=f"{nct_id}_{section}_00",
        score=0.8,
        text=f"Trial {nct_id}: Title\nSection: {section}\n{text}",
        metadata={"nct_id": nct_id, "section": section},
    )


class FakeRetriever:
    """Always returns the same two chunks."""

    def search(self, query, top_k=5):
        return [
            _result("NCT00000001", "eligibility", "Brain metastases are excluded."),
            _result("NCT00000002", "summary", "A lung cancer trial."),
        ][:top_k]


def test_extract_nct_ids_keeps_order_and_removes_duplicates() -> None:
    text = "See [NCT00000002] and [NCT00000001], also NCT00000002 again. Not NCT123."

    assert extract_nct_ids(text) == ["NCT00000002", "NCT00000001"]


def test_build_prompt_numbers_sources_and_includes_question() -> None:
    prompt = build_prompt("Which trials?", FakeRetriever().search("x"))

    assert "[1] (NCT00000001, eligibility)" in prompt
    assert "[2] (NCT00000002, summary)" in prompt
    assert "Brain metastases are excluded." in prompt
    assert "QUESTION: Which trials?" in prompt


def test_check_citations_flags_invented_ids() -> None:
    answer = "Excluded in [NCT00000001]; also see [NCT99999999]."

    check = check_citations(answer, FakeRetriever().search("x"))

    assert check["valid_citations"] == ["NCT00000001"]
    assert check["invented_citations"] == ["NCT99999999"]


def test_gold_coverage() -> None:
    assert gold_coverage(["NCT1", "NCT2"], ["NCT2", "NCT3"]) == {"found": 1, "total": 2}
    assert gold_coverage([], ["NCT2"]) == {"found": 0, "total": 0}


def test_answer_question_passes_sources_to_llm_and_checks_answer() -> None:
    received = {}

    def fake_llm(prompt, system_prompt):
        received["prompt"] = prompt
        received["system_prompt"] = system_prompt
        return "Trial [NCT00000001] excludes brain metastases."

    result = answer_question("Which trials exclude brain metastases?", FakeRetriever(), fake_llm)

    assert "Brain metastases are excluded." in received["prompt"]
    assert "ONLY" in received["system_prompt"]
    assert result["retrieved_nct_ids"] == ["NCT00000001", "NCT00000002"]
    assert result["valid_citations"] == ["NCT00000001"]
    assert result["invented_citations"] == []


def test_run_evaluation_saves_results_with_gold_scores(tmp_path) -> None:
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([
        {"id": "Q1", "category": "trial_search", "question": "Which trials?",
         "gold_nct_ids": ["NCT00000001", "NCT00000003"]},
    ]))
    output = tmp_path / "results.json"

    results = run_evaluation(
        FakeRetriever(),
        llm=lambda prompt, system_prompt: "See [NCT00000001].",
        questions_path=questions,
        results_path=output,
    )

    assert results[0]["gold_in_retrieved"] == {"found": 1, "total": 2}
    assert results[0]["gold_in_answer"] == {"found": 1, "total": 2}
    assert json.loads(output.read_text())[0]["id"] == "Q1"