"""BM25 keyword search over the trial chunks (the "lexical" half of hybrid search).

Dense search (bge + FAISS) matches MEANING; BM25 matches exact WORDS.
BM25 is strong for exact names that embeddings blur: drug codes (MK-3475, BL-B01D1),
trial IDs, gene names (EGFR, T790M), rare disease words.
"""
import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

CHUNKS_FILE = Path("data/processed/trial_chunks.jsonl")
CACHE_FILE = Path("data/index/bm25_trials.pkl")
TEXT_FIELDS = ("text", "chunk_text", "content")
ID_FIELDS = ("chunk_id", "id")
STOPWORDS = set("""a an and are as at be by for from has have in is it of on or that the this to was were
which with what who how does do any all find list show trials trial study studies patients""".split())


def tokenize(text):
    """'Osimertinib (AZD9291) in EGFR-mutant NSCLC' -> ['osimertinib', 'azd9291', 'egfr-mutant', 'nsclc']"""
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    return [w for w in words if w not in STOPWORDS]


def field(record, names):
    for name in names:
        if name in record:
            return record[name]
    raise KeyError(f"None of {names} found. Keys are: {list(record)}")


class BM25Index:
    def __init__(self, ids, texts):
        self.ids = ids
        self.texts = texts
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    @classmethod
    def from_chunks(cls, path=CHUNKS_FILE):
        ids, texts = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    ids.append(field(record, ID_FIELDS))
                    texts.append(field(record, TEXT_FIELDS))
        return cls(ids, texts)

    @classmethod
    def load_or_build(cls, cache=CACHE_FILE):
        """Building takes a little while, so keep a cached copy on disk."""
        if cache.exists():
            with open(cache, "rb") as f:
                return pickle.load(f)
        index = cls.from_chunks()
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(index, f)
        return index

    def search(self, query, top_k=50):
        """Return [(chunk_id, text, score)] best first."""
        scores = self.bm25.get_scores(tokenize(query))
        best = np.argsort(scores)[::-1][:top_k]
        return [(self.ids[i], self.texts[i], float(scores[i])) for i in best if scores[i] > 0]