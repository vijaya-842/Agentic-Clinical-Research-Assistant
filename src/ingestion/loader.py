import json
from pathlib import Path

from src.ingestion.document import Document


def load_json_documents(file_path: str | Path) -> list[Document]:
    """
    Load normalized documents from a JSON file.
    """

    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    documents = []

    for item in data:
        document = Document(
            id=item["id"],
            text=item["text"],
            source=item["source"],
            metadata=item.get("metadata", {}),
        )

        documents.append(document)

    return documents