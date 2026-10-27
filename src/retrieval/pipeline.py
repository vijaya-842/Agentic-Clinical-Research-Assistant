"""One place that builds the search stack, so every entry point uses the same setup.

Trials: AdvancedRetriever (filters) -> [HybridRetriever: + BM25, RRF] -> [MedCPT reranker]
Papers: PubMedRetriever (dense)     ->                                  [MedCPT reranker]
"""


def load_retrievers(hybrid=True, rerank=True):
    from src.reranking.cross_encoder import RerankingRetriever
    from src.retrieval.advanced import AdvancedRetriever
    from src.retrieval.pubmed_search import PubMedRetriever

    trials = AdvancedRetriever.from_disk()
    papers = PubMedRetriever.from_disk()
    if hybrid:
        from src.retrieval.bm25 import BM25Index
        from src.retrieval.hybrid import HybridRetriever
        trials = HybridRetriever(trials, BM25Index.load_or_build())
    if rerank:
        model = RerankingRetriever.load_model()
        trials = RerankingRetriever(trials, model)
        papers = RerankingRetriever(papers, model)
    print(f"Search: hybrid={'ON' if hybrid else 'OFF'}, reranker={'ON' if rerank else 'OFF'}")
    return trials, papers