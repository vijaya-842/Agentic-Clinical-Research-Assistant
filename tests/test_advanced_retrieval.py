import numpy as np
import pytest

from src.retrieval.advanced import (
    AdvancedRetriever,
    parse_filters,
    relaxation_steps,
    trial_matches,
)

VOCABULARY = ["lung", "breast", "brain"]


class FakeEncoder:
    """Turns text into 3 numbers: counts of 'lung', 'breast', 'brain'."""

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, **kwargs):
        vectors = np.array(
            [[t.lower().count(w) for w in VOCABULARY] for t in texts], dtype="float32"
        ) + 1e-6
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _chunk(chunk_id, text, status="RECRUITING", phases=("PHASE3",),
           conditions=("Lung Cancer",), interventions=("Osimertinib",)):
    return {
        "id": chunk_id,
        "text": text,
        "metadata": {
            "nct_id": chunk_id.split("_")[0],
            "title": "A trial",
            "status": status,
            "phases": list(phases),
            "conditions": list(conditions),
            "interventions": list(interventions),
        },
    }


CHUNKS = [
    _chunk("NCT00000001_summary_00", "lung lung trial"),
    _chunk("NCT00000001_eligibility_00", "lung brain metastases"),
    _chunk("NCT00000002_summary_00", "lung trial", status="COMPLETED"),
    _chunk("NCT00000003_summary_00", "lung trial", phases=("PHASE2",)),
    _chunk("NCT00000004_summary_00", "breast trial", conditions=("Breast Cancer",),
           interventions=("Pembrolizumab",)),
]


def _retriever():
    encoder = FakeEncoder()
    vectors = encoder.encode([c["text"] for c in CHUNKS])
    return AdvancedRetriever(CHUNKS, vectors, encoder)


def test_parse_filters_reads_status_phase_cancer_and_drugs() -> None:
    filters = parse_filters("Find recruiting Phase III NSCLC trials of osimertinib")

    assert filters["status"] == "RECRUITING"
    assert filters["phase"] == "PHASE3"
    assert "lung" in filters["cancer"]
    assert filters["drugs"] == ["osimertinib"]


def test_parse_filters_handles_numbers_and_not_yet_recruiting() -> None:
    filters = parse_filters("not yet recruiting phase 2 melanoma trials")

    assert filters["status"] == "NOT_YET_RECRUITING"
    assert filters["phase"] == "PHASE2"
    assert filters["cancer"] == ["melanoma"]


def test_parse_filters_finds_nothing_in_general_question() -> None:
    assert parse_filters("What is the mechanism of action of this drug?") == {}


def test_relaxation_drops_drugs_then_status_phase_then_cancer() -> None:
    steps = relaxation_steps({"status": "RECRUITING", "phase": "PHASE3",
                              "cancer": ["lung"], "drugs": ["xmab"]})

    assert steps[1] == {"status": "RECRUITING", "phase": "PHASE3", "cancer": ["lung"]}
    assert steps[2] == {"cancer": ["lung"]}
    assert steps[-1] == {}


def test_trial_matches() -> None:
    meta = CHUNKS[0]["metadata"]

    assert trial_matches(meta, {"status": "RECRUITING", "phase": "PHASE3"})
    assert trial_matches(meta, {"cancer": ["lung"], "drugs": ["osimertinib"]})
    assert not trial_matches(meta, {"status": "COMPLETED"})
    assert not trial_matches(meta, {"drugs": ["pembrolizumab"]})


def test_search_applies_filters() -> None:
    retriever = _retriever()

    results = retriever.search("recruiting phase 3 lung trials")

    assert {r.metadata["nct_id"] for r in results} == {"NCT00000001"}
    assert retriever.last_filters["relaxed"] is False


def test_search_returns_one_chunk_per_trial() -> None:
    results = _retriever().search("lung trials", top_k=5)

    nct_ids = [r.metadata["nct_id"] for r in results]
    assert len(nct_ids) == len(set(nct_ids))
    assert set(nct_ids) == {"NCT00000001", "NCT00000002", "NCT00000003"}


def test_search_relaxes_filters_when_nothing_matches() -> None:
    retriever = _retriever()

    results = retriever.search("recruiting lung trials of nivolumab")

    assert retriever.last_filters["relaxed"] is True
    assert "drugs" not in retriever.last_filters["applied"]
    assert results


def test_rejects_empty_query() -> None:
    with pytest.raises(ValueError):
        _retriever().search("  ")