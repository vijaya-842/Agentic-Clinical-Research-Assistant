from src.tools.count_trials import (
    answer_count_question,
    count_trials,
    is_count_question,
    parse_count_filters,
)

FAKE_TRIALS = [
    {"nct_id": "NCT001", "status": "RECRUITING", "phases": ["PHASE3"], "conditions": ["Pancreatic Cancer"]},
    {"nct_id": "NCT002", "status": "RECRUITING", "phases": ["PHASE2"], "conditions": ["Pancreatic Adenocarcinoma"]},
    {"nct_id": "NCT003", "status": "COMPLETED", "phases": ["PHASE3"], "conditions": ["Pancreatic Cancer"]},
    {"nct_id": "NCT004", "status": "RECRUITING", "phases": ["PHASE3"], "conditions": ["Breast Cancer"]},
    {"nct_id": "NCT005", "status": "NOT_YET_RECRUITING", "phases": [], "conditions": ["Pancreatic Cancer"]},
]


def test_detects_count_questions():
    assert is_count_question("How many recruiting lung cancer trials are there?")
    assert is_count_question("What is the number of phase 3 trials?")
    assert not is_count_question("Which trial tests thymosin alpha 1?")


def test_parses_status_phase_and_cancer():
    f = parse_count_filters("How many recruiting phase III pancreatic cancer trials?")
    assert f == {"status": "RECRUITING", "phase": "PHASE3", "cancer": "pancrea"}


def test_not_yet_recruiting_is_not_recruiting():
    f = parse_count_filters("How many not yet recruiting trials?")
    assert f["status"] == "NOT_YET_RECRUITING"


def test_counts_recruiting_pancreatic():
    f = parse_count_filters("How many recruiting pancreatic cancer trials?")
    count, ids = count_trials(FAKE_TRIALS, f)
    assert count == 2
    assert ids == ["NCT001", "NCT002"]


def test_answer_contains_the_number():
    result = answer_count_question("How many pancreatic cancer trials?", FAKE_TRIALS)
    assert result["count"] == 4
    assert "There are 4 matching trials" in result["answer"]