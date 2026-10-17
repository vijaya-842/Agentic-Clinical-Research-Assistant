import numpy as np

from src.embeddings.build_pubmed_index import build_index, paper_text
from src.retrieval.pubmed_search import QUERY_INSTRUCTION, PubMedRetriever

PAPERS = [
    {"pmid": "1", "title": "Osimertinib blocks EGFR", "abstract": "EGFR text", "year": "2020"},
    {"pmid": "2", "title": "Pembrolizumab blocks PD-1", "abstract": "PD-1 text", "year": "2021"},
]


class FakeModel:
    """Pretends to be the embedding model: 'EGFR' questions point one way, others the other way."""

    def __init__(self):
        self.last_input = None

    def encode(self, texts, normalize_embeddings=True):
        self.last_input = texts
        return np.array([[1.0, 0.0] if "EGFR" in t or "osimertinib" in t else [0.0, 1.0] for t in texts])


def make_retriever():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])  # paper 1 = EGFR, paper 2 = PD-1
    return PubMedRetriever(FakeModel(), build_index(vectors), PAPERS)


def test_paper_text_puts_title_first():
    assert paper_text(PAPERS[0]) == "Osimertinib blocks EGFR\nEGFR text"


def test_search_returns_best_paper_first():
    results = make_retriever().search("How does osimertinib work?", top_k=2)
    assert [r["pmid"] for r in results] == ["1", "2"]
    assert results[0]["score"] == 1.0


def test_question_gets_instruction_prefix():
    retriever = make_retriever()
    retriever.search("osimertinib", top_k=1)
    assert retriever.model.last_input == [QUERY_INSTRUCTION + "osimertinib"]


def test_top_k_larger_than_papers_is_safe():
    results = make_retriever().search("osimertinib", top_k=10)
    assert len(results) == 2