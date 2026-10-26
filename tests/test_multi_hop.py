from src.agents.multi_hop import answer_multi_hop, clean_drug, extract_drugs, is_multi_hop, trial_part

TRIALS = [
    {"nct_id": "NCT1", "interventions": ["Drug: Pembrolizumab 200 mg", "Placebo"]},
    {"nct_id": "NCT2", "interventions": ["Pembrolizumab", "Ipilimumab", "Radiation therapy"]},
]


def test_detects_multi_hop_questions():
    assert is_multi_hop("Which recruiting melanoma trials test pembrolizumab, and how does pembrolizumab work?")
    assert is_multi_hop("Find recruiting lung cancer trials and explain the drugs they test.")
    assert not is_multi_hop("What is the mechanism of action of osimertinib?")   # papers only
    assert not is_multi_hop("Find recruiting Phase 3 lung cancer trials.")       # trials only


def test_trial_part_drops_the_mechanism_half():
    q = "Which recruiting melanoma trials test pembrolizumab, and how does pembrolizumab work?"
    assert trial_part(q) == "Which recruiting melanoma trials test pembrolizumab"


def test_clean_drug():
    assert clean_drug("Drug: Osimertinib 80 mg") == "Osimertinib"


def test_extract_drugs_skips_placebo_and_radiation():
    assert extract_drugs(["NCT1", "NCT2"], TRIALS) == ["Pembrolizumab", "Ipilimumab"]


def test_combined_answer_and_citations():
    def trial_tool(q):
        return {"answer": "[NCT1] and [NCT2] test pembrolizumab.\n\nTrial details (from ClinicalTrials.gov data):\n- [NCT1] ...",
                "valid_citations": ["NCT1", "NCT2"], "invented_citations": [], "cited_nct_ids": ["NCT1", "NCT2"]}

    def paper_tool(q):
        return {"answer": "It blocks PD-1 [PMID 11111111].", "valid_citations": ["PMID 11111111"],
                "invented_citations": [], "cited_pmids": ["11111111"]}

    out = answer_multi_hop("Which trials test pembrolizumab and how does it work?", trial_tool, paper_tool, TRIALS)
    assert out["drugs"] == ["Pembrolizumab", "Ipilimumab"]
    assert "How the drugs work:" in out["answer"]
    assert out["answer"].index("How the drugs work:") < out["answer"].index("Trial details")
    assert out["valid_citations"] == ["NCT1", "NCT2", "PMID 11111111"]
    assert [h["step"] for h in out["hops"]] == ["trials", "papers", "papers"]