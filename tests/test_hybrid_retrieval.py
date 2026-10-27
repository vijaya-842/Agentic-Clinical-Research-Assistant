from src.evaluation.retrieval_metrics import hit_at_k, mrr, precision_at_k, recall_at_k
from src.retrieval.bm25 import BM25Index, tokenize
from src.retrieval.hybrid import rrf_fuse, to_trial_ranking


def test_tokenize_keeps_drug_codes_and_drops_stopwords():
    assert tokenize("Find trials of MK-3475 in EGFR-mutant NSCLC") == ["mk-3475", "egfr-mutant", "nsclc"]


def test_bm25_finds_exact_word():
    index = BM25Index(["NCT1_summary_00", "NCT2_summary_00", "NCT3_summary_00"],
                      ["Osimertinib for EGFR lung cancer", "Pembrolizumab in melanoma", "Breast cancer surgery"])
    assert index.search("pembrolizumab melanoma", top_k=3)[0][0] == "NCT2_summary_00"


def test_rrf_rewards_items_high_in_both_lists():
    assert rrf_fuse([["A", "B", "C"], ["B", "D", "A"]])[:2] == ["B", "A"]


def test_to_trial_ranking_dedupes_trials():
    assert to_trial_ranking(["NCT00000001_summary_00", "NCT00000001_eligibility_00", "NCT00000002_outcomes_00"]) == \
        ["NCT00000001", "NCT00000002"]


def test_metrics():
    ranked, gold = ["X", "G1", "Y", "G2"], ["G1", "G2", "G3"]
    assert recall_at_k(ranked, gold, 2) == 1 / 3
    assert precision_at_k(ranked, gold, 2) == 0.5
    assert hit_at_k(ranked, gold, 1) == 0.0
    assert mrr(ranked, gold) == 0.5