"""
Advanced retrieval, step 1:
  1. Read the question and detect filters (status, phase, cancer type, drug names).
  2. Search ONLY the chunks whose trial matches those filters.
  3. Keep the best chunk per trial, so 5 results = 5 different trials.
  4. If no trial matches, relax the filters step by step.

Try it from the project root:
    python -m src.retrieval.advanced "Find recruiting Phase 3 clinical trials in lung cancer"
"""

import re
import sys

import numpy as np

from src.retrieval.dense import DenseRetriever, SearchResult

# 1. Reading filters from the question (simple rules, easy to test and explain)

STATUS_WORDS = [
    ("not yet recruiting", "NOT_YET_RECRUITING"),
    ("recruiting", "RECRUITING"),
    ("completed", "COMPLETED"),
    ("terminated", "TERMINATED"),
]

PHASE_NUMBERS = {"i": "1", "ii": "2", "iii": "3", "iv": "4",
                 "1": "1", "2": "2", "3": "3", "4": "4"}
PHASE_PATTERN = re.compile(r"\bphase\s*(iv|iii|ii|i|[1-4])\b")

# Cancer words in a question -> words to look for in a trial's conditions/title.
CANCER_SYNONYMS = {
    "lung": ["lung", "nsclc"],
    "nsclc": ["lung", "nsclc"],
    "breast": ["breast"],
    "melanoma": ["melanoma"],
    "colorectal": ["colorectal", "colon", "rectal"],
    "colon": ["colorectal", "colon"],
    "prostate": ["prostate"],
    "pancreatic": ["pancrea"],
    "ovarian": ["ovarian", "ovary"],
    "leukemia": ["leukemia", "leukaemia"],
    "lymphoma": ["lymphoma"],
    "myeloma": ["myeloma"],
}

# Most targeted cancer drugs end in one of these (e.g. osimertinib, pembrolizumab).
DRUG_PATTERN = re.compile(r"\b[a-z]+(?:mab|nib|parib|ciclib|platin|taxel|rubicin)\b")


def parse_filters(question: str) -> dict:
    """Detect status, phase, cancer type and drug names in a question."""
    text = question.lower()
    filters: dict = {}

    for words, status in STATUS_WORDS:
        if words in text:
            filters["status"] = status
            break

    phase = PHASE_PATTERN.search(text)
    if phase:
        filters["phase"] = f"PHASE{PHASE_NUMBERS[phase.group(1)]}"

    cancer_words = []
    for word, synonyms in CANCER_SYNONYMS.items():
        if re.search(rf"\b{word}", text):
            cancer_words.extend(s for s in synonyms if s not in cancer_words)
    if cancer_words:
        filters["cancer"] = cancer_words

    drugs = list(dict.fromkeys(DRUG_PATTERN.findall(text)))
    if drugs:
        filters["drugs"] = drugs

    return filters


def relaxation_steps(filters: dict) -> list[dict]:
    """
    Filters to try in order, from strictest to none.
    If nothing matches, we drop drugs first, then status/phase, then cancer type.
    """
    steps = [dict(filters)]
    for keys in (["drugs"], ["status", "phase"], ["cancer"]):
        smaller = {k: v for k, v in steps[-1].items() if k not in keys}
        if smaller != steps[-1]:
            steps.append(smaller)
    if steps[-1]:
        steps.append({})
    return steps

# 2. Checking a chunk's trial against the filters

def trial_matches(metadata: dict, filters: dict) -> bool:
    if "status" in filters and metadata.get("status") != filters["status"]:
        return False
    if "phase" in filters and filters["phase"] not in (metadata.get("phases") or []):
        return False
    if "cancer" in filters:
        where = (" ".join(metadata.get("conditions") or []) + " "
                 + (metadata.get("title") or "")).lower()
        if not any(word in where for word in filters["cancer"]):
            return False
    if "drugs" in filters:
        interventions = " ".join(metadata.get("interventions") or []).lower()
        if not all(drug in interventions for drug in filters["drugs"]):
            return False
    return True

# 3. The retriever

class AdvancedRetriever:
    def __init__(self, chunks: list[dict], vectors: np.ndarray, encoder, query_instruction: str = ""):
        if len(chunks) != len(vectors):
            raise ValueError("Need exactly one vector per chunk.")
        self.chunks = chunks
        self.vectors = vectors  # vectors[i] belongs to chunks[i]
        self.encoder = encoder
        self.query_instruction = query_instruction
        self.last_filters: dict = {}

    @classmethod
    def from_disk(cls) -> "AdvancedRetriever":
        """Reuse the FAISS index we already built (no re-embedding needed)."""
        dense = DenseRetriever.from_disk()
        vectors = dense.index.reconstruct_n(0, dense.index.ntotal)
        return cls(dense.chunks, vectors, dense.encoder, dense.query_instruction)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty.")

        detected = parse_filters(query)

        # Find the strictest filters that still match at least one chunk.
        for filters in relaxation_steps(detected):
            positions = [i for i, chunk in enumerate(self.chunks)
                         if trial_matches(chunk["metadata"], filters)]
            if positions:
                break

        self.last_filters = {
            "detected": detected,
            "applied": filters,
            "relaxed": filters != detected,
            "matching_chunks": len(positions),
        }

        query_vector = np.asarray(
            self.encoder.encode([self.query_instruction + query],
                                normalize_embeddings=True, convert_to_numpy=True),
            dtype="float32",
        )[0]

        # Similarity with the allowed chunks only (dot product = cosine similarity).
        scores = self.vectors[positions] @ query_vector
        order = np.argsort(-scores)

        # Keep the best chunk of each trial.
        results, seen_trials = [], set()
        for rank in order:
            chunk = self.chunks[positions[rank]]
            nct_id = chunk["metadata"].get("nct_id")
            if nct_id in seen_trials:
                continue
            seen_trials.add(nct_id)
            results.append(SearchResult(chunk["id"], float(scores[rank]),
                                        chunk["text"], chunk["metadata"]))
            if len(results) == top_k:
                break
        return results


def main() -> None:
    query = " ".join(sys.argv[1:]) or "Find recruiting Phase 3 clinical trials in lung cancer"
    print("Loading index and model...")
    retriever = AdvancedRetriever.from_disk()

    results = retriever.search(query)
    print(f"\nQuestion: {query}")
    print(f"Filters: {retriever.last_filters}")
    for rank, result in enumerate(results, start=1):
        meta = result.metadata
        phases = ", ".join(meta.get("phases") or []) or "no phase"
        print(f"\n#{rank}  score={result.score:.3f}  {result.chunk_id}")
        print(f"    {meta.get('status')} | {phases} | {meta.get('url')}")
        print("    " + result.text[:250].replace("\n", " ") + "...")


if __name__ == "__main__":
    main()