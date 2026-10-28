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

def test_personal_advice_goes_to_safety_refusal():
    assert route("My father has stage IV lung cancer. Which treatment should he take?") == "safety_refusal"
    assert route("Should I take osimertinib?") == "safety_refusal"
    assert route("What should my mom do about her breast cancer?") == "safety_refusal"


def test_research_questions_are_not_refused():
    assert route("Which trials should patients with EGFR mutations consider?") != "safety_refusal"
    assert route("How many recruiting pancreatic cancer trials are there?") == "count_trials"

def test_trial_id_goes_to_lookup():
    assert route("Tell me about NCT05086614.") == "trial_lookup"


def test_guardrail_catches_like_me():
    assert route("Which immunotherapy would be best for a 70-year-old with stage III melanoma like me?") == "safety_refusal"

def test_greeting_goes_to_small_talk():
    assert route("hi") == "small_talk"
    assert route("hi, how many recruiting pancreatic cancer trials are there?") == "count_trials"

def test_multi_hop_route():
    assert route("Which recruiting melanoma trials test pembrolizumab, and how does pembrolizumab work?") == "multi_hop"
    assert route("What is the mechanism of action of osimertinib?") == "literature_rag"



def test_guardrail_catches_dose_changes():
    assert route("I have lung cancer and a new cough. Should I increase my medication?") == "safety_refusal"
    assert route("Should she skip her next infusion?") == "safety_refusal"


def test_list_adds_retrieved_trials_that_name_the_drug():
    from src.agents.router_agent import list_with_details
    trials = [
        {"nct_id": "NCT001", "title": "A", "status": "RECRUITING", "phases": [], "interventions": ["Drug: Azacitidine"]},
        {"nct_id": "NCT002", "title": "B", "status": "RECRUITING", "phases": [], "interventions": ["Venetoclax"]},
        {"nct_id": "NCT003", "title": "C", "status": "RECRUITING", "phases": [], "interventions": ["Placebo"]},
    ]
    result = {"answer": "Not enough information.", "valid_citations": [], "cited_nct_ids": [],
              "retrieved_nct_ids": ["NCT001", "NCT002", "NCT003"]}
    result = list_with_details(result, trials, "Which recruiting leukemia trials use azacitidine?")
    assert result["added_by_code"] == ["NCT001"]
    assert "NCT001" in result["answer"] and "NCT003" not in result["answer"]