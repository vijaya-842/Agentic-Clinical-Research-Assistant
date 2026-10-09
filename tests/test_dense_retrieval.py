import numpy as np
import pytest

from src.embeddings.build_faiss_index import build_index, embed_texts
from src.retrieval.dense import DenseRetriever, matches_filters


VOCABULARY = ["lung", "breast", "cake"]


class FakeEncoder:
    """
    A tiny stand-in for the real embedding model, so tests run instantly
    without downloading anything. Each text becomes 3 numbers: how often
    it mentions "lung", "breast" and "cake".
    """

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, **kwargs):
        vectors = np.array(
            [[text.lower().count(word) for word in VOCABULARY] for text in texts],
            dtype="float32",
        ) + 1e-6
        if normalize_embeddings:
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors


CHUNKS = [
    {
        "id": "NCT00000001_summary_00",
        "text": "lung lung trial",
        "metadata": {"status": "RECRUITING", "phases": ["PHASE3"], "section": "summary"},
    },
    {
        "id": "NCT00000002_summary_00",
        "text": "breast trial",
        "metadata": {"status": "COMPLETED", "phases": ["PHASE2"], "section": "summary"},
    },
    {
        "id": "NCT00000003_eligibility_00",
        "text": "lung eligibility",
        "metadata": {"status": "COMPLETED", "phases": ["PHASE2"], "section": "eligibility"},
    },
    {
        "id": "NCT00000004_summary_00",
        "text": "cake recipe",
        "metadata": {"status": "RECRUITING", "phases": [], "section": "summary"},
    },
]


def _retriever() -> DenseRetriever:
    encoder = FakeEncoder()
    vectors = embed_texts(encoder, [chunk["text"] for chunk in CHUNKS], batch_size=2)
    return DenseRetriever(build_index(vectors), CHUNKS, encoder)


def test_build_index_stores_one_vector_per_chunk() -> None:
    vectors = embed_texts(FakeEncoder(), [chunk["text"] for chunk in CHUNKS], batch_size=2)

    assert build_index(vectors).ntotal == len(CHUNKS)


def test_search_returns_most_similar_chunks_first() -> None:
    results = _retriever().search("lung", top_k=2)

    assert {result.chunk_id for result in results} == {
        "NCT00000001_summary_00",
        "NCT00000003_eligibility_00",
    }
    assert results[0].score >= results[1].score


def test_search_applies_metadata_filters() -> None:
    results = _retriever().search("lung", top_k=5, filters={"status": "RECRUITING"})

    assert results[0].chunk_id == "NCT00000001_summary_00"
    assert all(result.metadata["status"] == "RECRUITING" for result in results)


def test_search_filters_on_list_fields() -> None:
    results = _retriever().search("lung", top_k=5, filters={"phases": "PHASE2"})

    assert {result.chunk_id for result in results} == {
        "NCT00000002_summary_00",
        "NCT00000003_eligibility_00",
    }


def test_matches_filters() -> None:
    metadata = {"status": "RECRUITING", "phases": ["PHASE2", "PHASE3"]}

    assert matches_filters(metadata, {"phases": "PHASE3"})
    assert matches_filters(metadata, {"status": "RECRUITING", "phases": "PHASE2"})
    assert not matches_filters(metadata, {"status": "COMPLETED"})
    assert not matches_filters(metadata, {"phases": "PHASE1"})


def test_retriever_rejects_index_and_chunk_count_mismatch() -> None:
    encoder = FakeEncoder()
    vectors = embed_texts(encoder, [chunk["text"] for chunk in CHUNKS], batch_size=2)

    with pytest.raises(ValueError):
        DenseRetriever(build_index(vectors), CHUNKS[:3], encoder)


def test_search_rejects_empty_query() -> None:
    with pytest.raises(ValueError):
        _retriever().search("   ")