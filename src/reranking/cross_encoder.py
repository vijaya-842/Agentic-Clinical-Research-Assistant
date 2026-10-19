"""Reranker: re-orders search results by reading the question and each result TOGETHER.

FAISS (bi-encoder) compares two separate vectors -> fast but rough.
A cross-encoder reads "question + result" as one input -> slower but much more precise.
So: FAISS finds 20 candidates, the cross-encoder picks the best top_k.
"""
from dataclasses import is_dataclass

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # small (~90 MB), fast on CPU
CANDIDATES = 20

TEXT_FIELDS = ("text", "chunk_text", "content", "document")


def text_of(item):
    """Get the text to rerank from a search result (paper dict or trial chunk)."""
    if isinstance(item, dict):
        if "abstract" in item:  # PubMed paper
            return f"{item.get('title', '')}\n{item['abstract']}"
        for field in TEXT_FIELDS:
            if field in item:
                return item[field]
        raise KeyError(f"No text field found. Keys are: {list(item.keys())}")
    for field in TEXT_FIELDS:  # dataclass / object, e.g. SearchResult
        if hasattr(item, field):
            return getattr(item, field)
    fields = list(vars(item).keys()) if hasattr(item, "__dict__") or is_dataclass(item) else type(item)
    raise AttributeError(f"No text field found. Fields are: {fields}")


class RerankingRetriever:
    """Wraps any retriever that has .search(question, top_k=...).

    It asks the wrapped retriever for CANDIDATES results, scores each one with
    the cross-encoder, and returns the best top_k. Everything else (for example
    retriever.last_filters) is passed through to the wrapped retriever.
    """

    def __init__(self, base, model, candidates=CANDIDATES):
        self.base = base
        self.model = model
        self.candidates = candidates

    @classmethod
    def load_model(cls):
        from sentence_transformers import CrossEncoder
        return CrossEncoder(RERANKER_MODEL)

    def search(self, question, top_k=5, **kwargs):
        results = self.base.search(question, top_k=max(top_k, self.candidates), **kwargs)
        if len(results) <= 1:
            return results
        scores = self.model.predict([(question, text_of(r)) for r in results])
        ranked = sorted(zip(scores, range(len(results))), key=lambda pair: -pair[0])
        return [results[i] for _, i in ranked[:top_k]]

    def __getattr__(self, name):
        # called only for attributes RerankingRetriever does not have itself
        return getattr(self.base, name)


if __name__ == "__main__":
    # Quick before/after demo on the PubMed papers
    from src.retrieval.pubmed_search import PubMedRetriever

    question = "What is the mechanism of action of osimertinib?"
    papers = PubMedRetriever.from_disk()
    reranked = RerankingRetriever(papers, RerankingRetriever.load_model())

    print("Question:", question)
    print("\nBEFORE (FAISS only):")
    for p in papers.search(question, top_k=5):
        print(f"  PMID {p['pmid']}  {p['title'][:90]}")
    print("\nAFTER (FAISS top 20 -> reranker top 5):")
    for p in reranked.search(question, top_k=5):
        print(f"  PMID {p['pmid']}  {p['title'][:90]}")