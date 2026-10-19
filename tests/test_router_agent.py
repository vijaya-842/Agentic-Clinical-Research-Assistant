from src.agents.router_agent import answer_with_count_tool, coverage, route, trial_details

FAKE_TRIALS = [
    {"nct_id": "NCT001", "status": "RECRUITING", "phases": ["PHASE3"], "conditions": ["Pancreatic Cancer"]},
    {"nct_id": "NCT002", "status": "COMPLETED", "phases": ["PHASE2"], "conditions": ["Pancreatic Cancer"]},
]


def test_count_question_goes_to_count_tool():
    assert route("How many recruiting pancreatic cancer trials are there?") == "count_trials"


def test_other_questions_go_to_rag():
    assert route("Which trial tests thymosin alpha 1 in colorectal cancer?") == "advanced_rag"

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

def test_list_questions_go_to_list_rag():
    assert route("List clinical trials that include osimertinib as an intervention.") == "list_rag"
    assert route("Find recruiting Phase 3 clinical trials in lung cancer.") == "list_rag"
    assert route("Which non-small cell lung cancer trials exclude patients with brain metastases?") == "list_rag"


def test_single_trial_question_is_not_a_list():
    assert route("Which trial combines capecitabine and tucatinib?") == "advanced_rag"

FAKE_TRIALS_WITH_TITLES = [
    {"nct_id": "NCT001", "title": "A Study of Drug X", "status": "RECRUITING", "phases": ["PHASE3"]},
]


def test_trial_details_lists_status_and_phase():
    text = trial_details(["NCT001"], FAKE_TRIALS_WITH_TITLES)
    assert "[NCT001]" in text
    assert "Status: RECRUITING" in text
    assert "Phase: Phase 3" in text


def test_trial_details_skips_unknown_ids():
    assert trial_details(["NCT999"], FAKE_TRIALS_WITH_TITLES) == ""

def test_literature_questions_go_to_literature_rag():
    assert route("What is the mechanism of action of osimertinib?") == "literature_rag"
    assert route("How does pembrolizumab work?") == "literature_rag"
    assert route("What are the side effects of trastuzumab?") == "literature_rag"
