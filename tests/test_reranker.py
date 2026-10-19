from dataclasses import dataclass

from src.reranking.cross_encoder import RerankingRetriever, text_of


@dataclass
class Chunk:
    nct_id: str
    text: str


class FakeBase:
    """Returns results in a deliberately WRONG order, and remembers top_k."""

    def __init__(self, results):
        self.results = results
        self.asked_top_k = None
        self.last_filters = {"status": "RECRUITING"}

    def search(self, question, top_k=5, **kwargs):
        self.asked_top_k = top_k
        return self.results[:top_k]


class FakeCrossEncoder:
    """Scores a result high if it mentions 'mechanism of action'."""

    def predict(self, pairs):
        return [1.0 if "mechanism of action" in text else 0.0 for _, text in pairs]


PAPERS = [
    {"pmid": "1", "title": "Resistance to osimertinib", "abstract": "Resistance mechanisms."},
    {"pmid": "2", "title": "Osimertinib review", "abstract": "Its mechanism of action is EGFR inhibition."},
]


def test_text_of_paper_and_chunk():
    assert text_of(PAPERS[0]) == "Resistance to osimertinib\nResistance mechanisms."
    assert text_of(Chunk("NCT1", "chunk text")) == "chunk text"


def test_reranker_moves_best_result_to_top():
    retriever = RerankingRetriever(FakeBase(PAPERS), FakeCrossEncoder())
    results = retriever.search("How does osimertinib work?", top_k=1)
    assert [r["pmid"] for r in results] == ["2"]


def test_reranker_asks_base_for_more_candidates():
    base = FakeBase(PAPERS)
    RerankingRetriever(base, FakeCrossEncoder(), candidates=20).search("q", top_k=5)
    assert base.asked_top_k == 20


def test_other_attributes_pass_through():
    retriever = RerankingRetriever(FakeBase(PAPERS), FakeCrossEncoder())
    assert retriever.last_filters == {"status": "RECRUITING"}


def test_works_with_trial_chunks():
    chunks = [Chunk("NCT1", "unrelated"), Chunk("NCT2", "the mechanism of action")]
    results = RerankingRetriever(FakeBase(chunks), FakeCrossEncoder()).search("q", top_k=1)
    assert results[0].nct_id == "NCT2"