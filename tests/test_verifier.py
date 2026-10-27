from src.agents.verifier import citation_coverage, status_issues, verify_answer

TRIALS = [
    {"nct_id": "NCT02000596", "status": "TERMINATED"},
    {"nct_id": "NCT07642024", "status": "RECRUITING"},
    {"nct_id": "NCT07680790", "status": "NOT_YET_RECRUITING"},
]
BY_ID = {t["nct_id"]: t for t in TRIALS}


def result(answer, invented=()):
    return {"answer": answer, "invented_citations": list(invented), "valid_citations": []}


def test_flags_terminated_trial_called_open():
    issues = status_issues("[TRIAL NCT02000596] This trial is open to new patients.", BY_ID)
    assert len(issues) == 1 and "TERMINATED" in issues[0]


def test_flags_not_yet_recruiting_called_recruiting():
    text = "[NCT07642024] and [NCT07680790] are currently recruiting patients."
    issues = status_issues(text, BY_ID)
    assert len(issues) == 1 and "NCT07680790" in issues[0]


def test_negated_claims_are_not_flagged():
    assert status_issues("[NCT02000596] is not recruiting patients.", BY_ID) == []
    assert status_issues("[NCT02000596] is ACTIVE_NOT_RECRUITING.", BY_ID) == []


def test_code_written_details_are_not_checked():
    answer = ("[NCT07642024] is recruiting.\n\nTrial details (from ClinicalTrials.gov data):\n"
              "- [NCT07680790] Title | Status: NOT_YET_RECRUITING | Phase: Phase 3")
    out = verify_answer(result(answer), TRIALS, "list_trials")
    assert out["verification"]["status"] == "pass"


def test_warning_adds_note_to_answer():
    out = verify_answer(result("[NCT02000596] is open to new patients."), TRIALS, "search_trials")
    assert out["verification"]["status"] == "warning"
    assert "Verifier note" in out["answer"]


def test_invented_citation_is_an_issue():
    out = verify_answer(result("Some claim [PMID 99999999].", invented=["PMID 99999999"]), TRIALS, "search_papers")
    assert out["verification"]["status"] == "warning"


def test_code_tools_are_skipped():
    out = verify_answer(result("There are 68 matching trials."), TRIALS, "count_trials")
    assert out["verification"]["status"] == "skipped"


def test_citation_coverage():
    text = "Osimertinib blocks mutant EGFR signalling [PMID 37060646]. It is taken orally once daily by patients."
    assert citation_coverage(text) == 0.5

def test_low_citation_coverage_is_a_warning():
    text = ("[NCT07642024] tests pembrolizumab in melanoma patients. "
            "Pembrolizumab works by targeting PD-1 on immune cells. "
            "It helps immune cells attack the cancer cells.")
    out = verify_answer(result(text), TRIALS, "search_trials")
    assert out["verification"]["status"] == "warning"
    assert "cite a source" in out["verification"]["issues"][0]



def test_citation_after_full_stop_counts():
    text = ("Osimertinib is a third-generation EGFR-TKI that irreversibly binds EGFR activating mutations "
            "and T790M, while sparing wild-type EGFR. [PMID 37060646]")
    assert citation_coverage(text) == 1.0


def test_structured_block_with_id_on_its_own_line_counts():
    text = "[1] NCT05086614\nSponsor: Fudan University\nPrimary outcome: 3-year disease-free survival rate"
    assert citation_coverage(text) == 1.0