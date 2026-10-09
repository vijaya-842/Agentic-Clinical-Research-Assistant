"""
Embed every trial chunk and store the vectors in a FAISS index.

Run from the project root (about 20-40 minutes on a laptop CPU):
    python -m src.embeddings.build_faiss_index
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "trial_chunks.jsonl"
INDEX_PATH = PROCESSED_DIR / "trial_chunks.faiss"
IDS_PATH = PROCESSED_DIR / "trial_chunks_ids.json"
MANIFEST_PATH = PROCESSED_DIR / "trial_chunks_index_manifest.json"
MODELS_CONFIG = PROJECT_ROOT / "configs" / "models.yaml"


def load_embedding_config() -> dict:
    """Read the embedding section of configs/models.yaml."""
    with open(MODELS_CONFIG, encoding="utf-8") as file:
        config = yaml.safe_load(file)["embedding"]
    if not config.get("model"):
        raise ValueError("Set embedding.model in configs/models.yaml first.")
    return config


def load_chunks(path: Path) -> list[dict]:
    """Read the chunks file: one JSON object per line."""
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def embed_texts(encoder, texts: list[str], batch_size: int) -> np.ndarray:
    """
    Turn texts into vectors.

    normalize_embeddings=True makes every vector length 1, so the
    dot product of two vectors equals their cosine similarity.
    """
    vectors = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype="float32")


def build_index(vectors: np.ndarray) -> faiss.Index:
    """
    IndexFlatIP compares a question with EVERY stored vector using the
    inner (dot) product. Exact, and fast enough for tens of thousands of chunks.
    """
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(index: faiss.Index, ids: list[str], index_path: Path, ids_path: Path) -> None:
    """
    FAISS stores only numbers, so we also save which chunk each vector is:
    ids[0] is the chunk for vector 0, ids[1] for vector 1, and so on.
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(ids_path, "w", encoding="utf-8") as file:
        json.dump(ids, file)


def run() -> None:
    # Imported here because it is slow to import and not needed by the tests.
    from sentence_transformers import SentenceTransformer

    config = load_embedding_config()
    batch_size = config.get("batch_size", 32)

    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")
    print(f"Embedding model: {config['model']} (batch size {batch_size})")

    encoder = SentenceTransformer(config["model"])

    start = time.time()
    vectors = embed_texts(encoder, [chunk["text"] for chunk in chunks], batch_size)
    minutes = (time.time() - start) / 60
    print(f"Embedded {len(vectors)} chunks in {minutes:.1f} minutes")

    index = build_index(vectors)
    save_index(index, [chunk["id"] for chunk in chunks], INDEX_PATH, IDS_PATH)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": config["model"],
        "vector_dimension": int(vectors.shape[1]),
        "chunks": len(chunks),
        "index_type": "IndexFlatIP (cosine similarity on normalized vectors)",
        "embedding_minutes": round(minutes, 1),
        "chunks_file": str(CHUNKS_PATH.relative_to(PROJECT_ROOT)),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print(f"Saved index to {INDEX_PATH}")
    print(f"Saved chunk ids to {IDS_PATH}")
    print(f"Saved manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    run()