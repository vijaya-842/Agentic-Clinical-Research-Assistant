from src.tools.trial_lookup import answer_trial_lookup, find_nct_ids

TRIALS = [
    {"nct_id": "NCT05086614", "title": "Thymosin Alpha 1 in Colorectal Cancer", "sponsor": "Fudan University",
     "status": "RECRUITING", "phases": ["PHASE3"], "conditions": ["Colorectal Cancer"],
     "interventions": ["Thymosin alpha 1"], "primary_outcomes": [{"measure": "3-year DFS rate"}],
     "summary": "A trial of thymosin."},
]


def test_find_nct_ids():
    assert find_nct_ids("Tell me about nct05086614 and NCT05086614") == ["NCT05086614"]
    assert find_nct_ids("No IDs here") == []


def test_lookup_known_trial():
    result = answer_trial_lookup("Tell me about NCT05086614.", TRIALS)
    assert "Fudan University" in result["answer"]
    assert "3-year DFS rate" in result["answer"]
    assert result["valid_citations"] == ["NCT05086614"]


def test_lookup_unknown_trial():
    result = answer_trial_lookup("What is NCT99999999?", TRIALS)
    assert "Not in our dataset: NCT99999999" in result["answer"]
    assert result["valid_citations"] == []