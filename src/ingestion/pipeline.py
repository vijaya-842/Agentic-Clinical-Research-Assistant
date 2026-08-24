from pathlib import Path
import json
import sys

import yaml


# Find the project root directory.
project_root = Path(__file__).resolve().parents[2]

# Allow imports from the project root.
sys.path.append(str(project_root))

from src.ingestion.loader import load_json_documents
from src.ingestion.chunker import chunk_documents


def load_retrieval_config() -> dict:
    """
    Load chunking and retrieval configuration.
    """

    config_path = project_root / "configs" / "retrieval.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def save_processed_documents(documents, output_path: Path) -> None:
    """
    Save processed document chunks as JSON.
    """

    data = []

    for document in documents:
        data.append(
            {
                "id": document.id,
                "text": document.text,
                "source": document.source,
                "metadata": document.metadata,
            }
        )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def run_ingestion(input_file: str, output_file: str) -> None:
    """
    Load raw documents, chunk them, and save processed chunks.
    """

    config = load_retrieval_config()

    chunk_size = config["chunking"]["chunk_size"]
    chunk_overlap = config["chunking"]["chunk_overlap"]

    input_path = project_root / input_file
    output_path = project_root / output_file

    print(f"\nLoading documents from: {input_path}")

    documents = load_json_documents(input_path)

    print(f"Loaded {len(documents)} documents.")

    print("\nChunking documents...")
    print(f"Chunk size: {chunk_size}")
    print(f"Chunk overlap: {chunk_overlap}")

    chunked_documents = chunk_documents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    print(f"Created {len(chunked_documents)} chunks.")

    save_processed_documents(
        documents=chunked_documents,
        output_path=output_path,
    )

    print(f"\nProcessed documents saved to: {output_path}")


if __name__ == "__main__":

    run_ingestion(
        input_file="data/raw/sample_biomedical_documents.json",
        output_file="data/processed/sample_biomedical_chunks.json",
    )