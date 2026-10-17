"""Build a FAISS index for the PubMed papers (1 paper = 1 card).

Input : data/raw/pubmed_papers.json
Output: data/index/pubmed.faiss        (the vectors)
        data/index/pubmed_papers.json  (the papers, in the same order as the vectors)
"""
import json
import time
from pathlib import Path

import faiss
import numpy as np

PAPERS_FILE = Path("data/raw/pubmed_papers.json")
INDEX_DIR = Path("data/index")
INDEX_FILE = INDEX_DIR / "pubmed.faiss"
PAPERS_OUT = INDEX_DIR / "pubmed_papers.json"

MODEL_NAME = "BAAI/bge-base-en-v1.5"  # same model as the trial index (configs/models.yaml)
BATCH_SIZE = 32


def paper_text(paper):
    """The text we embed: title first (so it is never cut off), then the abstract."""
    return f"{paper['title']}\n{paper['abstract']}"


def load_papers(path=PAPERS_FILE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_index(vectors):
    """Inner-product index; with normalized vectors this is cosine similarity."""
    vectors = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def main():
    from sentence_transformers import SentenceTransformer

    papers = load_papers()
    print(f"Papers: {len(papers)}")
    model = SentenceTransformer(MODEL_NAME)

    start = time.time()
    vectors = model.encode(
        [paper_text(p) for p in papers],
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"Embedded in {(time.time() - start) / 60:.1f} minutes, shape {vectors.shape}")

    index = build_index(vectors)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    with open(PAPERS_OUT, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False)
    print(f"Saved {index.ntotal} vectors to {INDEX_FILE}")


if __name__ == "__main__":
    main()