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