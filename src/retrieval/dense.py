"""
Dense (embedding) search over the trial chunks FAISS index.

Try it from the project root:
    python -m src.retrieval.dense "NSCLC trials that exclude patients with brain metastases"
"""

import json
import sys
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
import yaml

from src.embeddings.build_faiss_index import (
    CHUNKS_PATH,
    IDS_PATH,
    INDEX_PATH,
    PROJECT_ROOT,
    load_chunks,
    load_embedding_config,
)


@dataclass
class SearchResult:
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


def matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    """
    Check a chunk's metadata against simple filters, for example
    {"status": "RECRUITING", "phases": "PHASE3", "section": "eligibility"}.

    For list fields such as phases or conditions, the wanted value
    must be one of the items in the list.
    """
    for key, wanted in filters.items():
        value = metadata.get(key)
        if isinstance(value, list):
            if wanted not in value:
                return False
        elif value != wanted:
            return False
    return True


class DenseRetriever:
    def __init__(self, index: faiss.Index, chunks: list[dict], encoder, query_instruction: str = ""):
        if index.ntotal != len(chunks):
            raise ValueError(
                f"Index has {index.ntotal} vectors but {len(chunks)} chunks were given."
            )
        self.index = index
        self.chunks = chunks  # chunks[i] belongs to vector i
        self.encoder = encoder
        self.query_instruction = query_instruction

    @classmethod
    def from_disk(cls) -> "DenseRetriever":
        """Load the saved index, the chunks and the embedding model."""
        from sentence_transformers import SentenceTransformer

        config = load_embedding_config()
        index = faiss.read_index(str(INDEX_PATH))

        with open(IDS_PATH, encoding="utf-8") as file:
            ids = json.load(file)
        chunks_by_id = {chunk["id"]: chunk for chunk in load_chunks(CHUNKS_PATH)}
        chunks = [chunks_by_id[chunk_id] for chunk_id in ids]

        return cls(
            index=index,
            chunks=chunks,
            encoder=SentenceTransformer(config["model"]),
            query_instruction=config.get("query_instruction", ""),
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        candidates: int = 200,
    ) -> list[SearchResult]:
        """
        Return the top_k chunks most similar to the query.

        With filters, we first take the `candidates` most similar chunks and
        then keep only those that match. (FAISS cannot filter by itself;
        Qdrant can, which we may switch to later.)
        """
        if not query.strip():
            raise ValueError("query must not be empty.")

        query_vector = self.encoder.encode(
            [self.query_instruction + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        query_vector = np.asarray(query_vector, dtype="float32")

        k = candidates if filters else top_k
        k = min(k, self.index.ntotal)
        scores, positions = self.index.search(query_vector, k)

        results = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:  # FAISS uses -1 for "no result"
                continue
            chunk = self.chunks[position]
            if filters and not matches_filters(chunk["metadata"], filters):
                continue
            results.append(
                SearchResult(
                    chunk_id=chunk["id"],
                    score=float(score),
                    text=chunk["text"],
                    metadata=chunk["metadata"],
                )
            )
            if len(results) == top_k:
                break
        return results


def main() -> None:
    query = " ".join(sys.argv[1:]) or (
        "NSCLC trials that exclude patients with brain metastases"
    )

    with open(PROJECT_ROOT / "configs" / "retrieval.yaml", encoding="utf-8") as file:
        top_k = yaml.safe_load(file)["retrieval"]["top_k"]

    print("Loading index and model...")
    retriever = DenseRetriever.from_disk()

    print(f"\nQuestion: {query}")
    for rank, result in enumerate(retriever.search(query, top_k=top_k), start=1):
        meta = result.metadata
        phases = ", ".join(meta.get("phases") or []) or "no phase"
        print(f"\n#{rank}  score={result.score:.3f}  {result.chunk_id}")
        print(f"    {meta.get('status')} | {phases} | {meta.get('url')}")
        print("    " + result.text[:300].replace("\n", " ") + "...")


if __name__ == "__main__":
    main()