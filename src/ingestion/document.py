from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """
    Represents one normalized document used by the RAG pipeline.
    """

    id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)