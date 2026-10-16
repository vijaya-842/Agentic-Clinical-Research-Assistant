from src.agents.router_agent import answer_with_count_tool, coverage, route

FAKE_TRIALS = [
    {"nct_id": "NCT001", "status": "RECRUITING", "phases": ["PHASE3"], "conditions": ["Pancreatic Cancer"]},
    {"nct_id": "NCT002", "status": "COMPLETED", "phases": ["PHASE2"], "conditions": ["Pancreatic Cancer"]},
]


def test_count_question_goes_to_count_tool():
    assert route("How many recruiting pancreatic cancer trials are there?") == "count_trials"


def test_other_questions_go_to_rag():
    assert route("Which trial tests thymosin alpha 1 in colorectal cancer?") == "advanced_rag"
    assert route("What is the mechanism of action of osimertinib?") == "advanced_rag"


def test_count_tool_result_has_same_keys_as_rag():
    result = answer_with_count_tool("How many recruiting pancreatic trials?", FAKE_TRIALS)
    for key in ["question", "answer", "retrieved_nct_ids", "cited_nct_ids",
                "valid_citations", "invented_citations", "filters"]:
        assert key in result
    assert "There are 1 matching" in result["answer"]
    assert result["invented_citations"] == []


def test_coverage():
    assert coverage(["NCT1", "NCT2"], ["NCT2", "NCT9"]) == {"found": 1, "total": 2}
    assert coverage([], ["NCT1"]) == {"found": 0, "total": 0}