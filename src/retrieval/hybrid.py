"""Hybrid search: combine the dense ranking and the BM25 ranking with Reciprocal Rank Fusion.

RRF score of an item = sum over rankings of 1 / (k + rank).
Items ranked high by EITHER method rise to the top; items ranked high by BOTH rise most.
RRF only uses ranks, so the two very different score scales never need to be compared.
"""
import re

NCT = re.compile(r"NCT\d{8}")


def rrf_fuse(rankings, k=60):
    """rankings: lists of ids, best first. Returns ids ordered by fused score."""
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item: -scores[item])


def to_trial_ranking(chunk_ids_or_texts):
    """Chunk ranking -> trial ranking: keep each NCT ID the first time it appears."""
    trials = []
    for item in chunk_ids_or_texts:
        match = NCT.search(str(item))
        if match and match.group() not in trials:
            trials.append(match.group())
    return trials

class HybridRetriever:
    """Dense (filtered) + BM25 (same filters), fused with RRF. One result per trial.

    Wraps the AdvancedRetriever, so the SAME filters (status, phase, cancer, drugs)
    and the same filter relaxation are used for both halves.
    """

    def __init__(self, base, bm25, candidates=50, bm25_pool=300, k=60):
        self.base = base            # AdvancedRetriever
        self.bm25 = bm25            # BM25Index over the same chunks
        self.candidates = candidates
        self.bm25_pool = bm25_pool  # BM25 chunks to look at before filtering
        self.k = k
        self.position = {chunk["id"]: i for i, chunk in enumerate(base.chunks)}

    def bm25_results(self, query, filters):
        from src.retrieval.advanced import trial_matches
        from src.retrieval.dense import SearchResult

        results, seen = [], set()
        for chunk_id, _, score in self.bm25.search(query, top_k=self.bm25_pool):
            pos = self.position.get(chunk_id)
            if pos is None:
                continue
            chunk = self.base.chunks[pos]
            nct = chunk["metadata"].get("nct_id")
            if nct in seen or not trial_matches(chunk["metadata"], filters):
                continue
            seen.add(nct)
            results.append(SearchResult(chunk["id"], score, chunk["text"], chunk["metadata"]))
            if len(results) == self.candidates:
                break
        return results

    def search(self, query, top_k=5):
        dense = self.base.search(query, top_k=max(top_k, self.candidates))  # also sets last_filters
        lexical = self.bm25_results(query, self.base.last_filters.get("applied", {}))

        by_trial = {}
        for result in lexical + dense:  # dense last, so its best chunk is kept when both found a trial
            by_trial[result.metadata.get("nct_id")] = result
        order = rrf_fuse([[r.metadata.get("nct_id") for r in dense],
                          [r.metadata.get("nct_id") for r in lexical]], k=self.k)
        return [by_trial[nct] for nct in order[:top_k]]

    def __getattr__(self, name):
        return getattr(self.base, name)  # e.g. last_filters, chunks