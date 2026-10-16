import json

from src.rag.advanced_rag import ADVANCED_SYSTEM_PROMPT, answer_question, run_evaluation
from src.rag.trial_facts import build_prompt_with_facts, format_fact_sheet, load_trial_facts
from src.retrieval.dense import SearchResult


FACTS = {
    "NCT00000001": {
        "title": "Drug X after surgery",
        "sponsor": "Fudan University",
        "status": "RECRUITING",
        "phases": ["PHASE3"],
        "conditions": ["Colorectal Cancer"],
        "interventions": ["Drug X"],
        "primary_outcomes": ["3-year disease-free survival rate"],
    },
}


def _result(nct_id, section="summary", text="some card text"):
    return SearchResult(f"{nct_id}_{section}_00", 0.8, text,
                        {"nct_id": nct_id, "section": section})


class FakeRetriever:
    last_filters = {"applied": {"cancer": ["colorectal"]}, "relaxed": False}

    def search(self, query, top_k=5):
        return [_result("NCT00000001"), _result("NCT00000002", "outcomes", "ctDNA positivity")]


def test_load_trial_facts_keys_by_nct_id(tmp_path) -> None:
    path = tmp_path / "trials.json"
    path.write_text(json.dumps([
        {"id": "NCT00000001", "text": "...", "source": "x", "metadata": FACTS["NCT00000001"]},
    ]))

    facts = load_trial_facts(path)

    assert facts["NCT00000001"]["sponsor"] == "Fudan University"


def test_format_fact_sheet_contains_key_details() -> None:
    sheet = format_fact_sheet("NCT00000001", FACTS["NCT00000001"])

    assert "Sponsor: Fudan University" in sheet
    assert "Status: RECRUITING" in sheet
    assert "Phase: PHASE3" in sheet
    assert "Primary outcomes: 3-year disease-free survival rate" in sheet


def test_format_fact_sheet_shortens_long_lists() -> None:
    sheet = format_fact_sheet("NCT1", {"interventions": [f"drug{i}" for i in range(8)]})

    assert "drug4" in sheet
    assert "drug5" not in sheet
    assert "(8 in total)" in sheet


def test_prompt_puts_each_fact_sheet_with_its_own_trial() -> None:
    prompt = build_prompt_with_facts("Which trial?", FakeRetriever().search("x"), FACTS)

    first, second = prompt.split("[2] Trial NCT00000002")
    assert "Primary outcomes: 3-year disease-free survival rate" in first
    assert "(no fact sheet for NCT00000002)" in second
    assert "ctDNA positivity" in second
    assert "QUESTION: Which trial?" in prompt


def test_answer_question_uses_fact_sheets_and_advanced_rules() -> None:
    received = {}

    def fake_llm(prompt, system_prompt):
        received["prompt"], received["system"] = prompt, system_prompt
        return "Sponsored by Fudan University [NCT00000001]."

    result = answer_question("Which trial?", FakeRetriever(), FACTS, fake_llm)

    assert "Sponsor: Fudan University" in received["prompt"]
    assert received["system"] == ADVANCED_SYSTEM_PROMPT
    assert "Never take a detail from one trial" in ADVANCED_SYSTEM_PROMPT
    assert result["valid_citations"] == ["NCT00000001"]
    assert result["filters"]["applied"] == {"cancer": ["colorectal"]}


def test_run_evaluation_saves_results(tmp_path) -> None:
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps([
        {"id": "Q1", "category": "trial_lookup", "question": "Which trial?",
         "gold_nct_ids": ["NCT00000001"]},
    ]))
    output = tmp_path / "out.json"

    results = run_evaluation(FakeRetriever(), FACTS,
                             llm=lambda prompt, system_prompt: "[NCT00000001]",
                             questions_path=questions, results_path=output)

    assert results[0]["gold_in_answer"] == {"found": 1, "total": 1}
    assert json.loads(output.read_text())[0]["id"] == "Q1"