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

class FakeAdvanced:
    """Pretends to be AdvancedRetriever: dense finds only NCT1; filters require RECRUITING."""

    def __init__(self):
        self.chunks = [
            {"id": "NCT1_summary_00", "text": "Trial NCT1: pembrolizumab melanoma", "metadata": {"nct_id": "NCT1", "status": "RECRUITING"}},
            {"id": "NCT2_summary_00", "text": "Trial NCT2: MK-3475 melanoma", "metadata": {"nct_id": "NCT2", "status": "RECRUITING"}},
            {"id": "NCT3_summary_00", "text": "Trial NCT3: MK-3475 melanoma", "metadata": {"nct_id": "NCT3", "status": "COMPLETED"}},
        ]
        self.last_filters = {}

    def search(self, query, top_k=5):
        from src.retrieval.dense import SearchResult
        self.last_filters = {"applied": {"status": "RECRUITING"}}
        c = self.chunks[0]
        return [SearchResult(c["id"], 0.9, c["text"], c["metadata"])]


def test_hybrid_adds_bm25_trials_that_pass_the_same_filters():
    from src.retrieval.hybrid import HybridRetriever
    base = FakeAdvanced()
    # a few unrelated filler chunks so BM25 word weights behave like in a real corpus
    filler = [("F1", "breast cancer surgery"), ("F2", "prostate radiation therapy"),
              ("F3", "colon cancer screening"), ("F4", "leukemia stem cell transplant")]
    bm25 = BM25Index([c["id"] for c in base.chunks] + [f[0] for f in filler],
                     [c["text"] for c in base.chunks] + [f[1] for f in filler])
    results = HybridRetriever(base, bm25).search("MK-3475 melanoma", top_k=5)
    ids = [r.metadata["nct_id"] for r in results]
    assert "NCT2" in ids          # found by BM25 only (exact code MK-3475)
    assert "NCT3" not in ids      # BM25 found it too, but it fails the RECRUITING filter
    assert "NCT1" in ids          # found by dense