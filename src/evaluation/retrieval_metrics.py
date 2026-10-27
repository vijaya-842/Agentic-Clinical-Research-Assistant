"""Standard retrieval metrics. `ranked` = retrieved IDs best first, `gold` = the correct IDs."""


def recall_at_k(ranked, gold, k):
    """Share of the gold items found in the top k."""
    return len(set(ranked[:k]) & set(gold)) / len(gold) if gold else None


def precision_at_k(ranked, gold, k):
    """Share of the top k that are gold items."""
    top = ranked[:k]
    return len(set(top) & set(gold)) / len(top) if top else 0.0


def hit_at_k(ranked, gold, k):
    """1 if at least one gold item is in the top k, else 0."""
    return 1.0 if set(ranked[:k]) & set(gold) else 0.0


def mrr(ranked, gold):
    """1 / rank of the first gold item (0 if none). Rewards putting a correct item first."""
    for rank, item in enumerate(ranked, start=1):
        if item in gold:
            return 1.0 / rank
    return 0.0


def average(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None