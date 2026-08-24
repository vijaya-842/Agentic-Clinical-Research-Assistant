from src.ingestion.document import Document


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[str]:
    """
    Split text into overlapping chunks without cutting words.
    """

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    # Normalize spaces and newlines.
    text = " ".join(text.split())

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(start + chunk_size, text_length)

        # Try to end at a space instead of cutting a word.
        if end < text_length:
            last_space = text.rfind(" ", start, end)

            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Stop if we reached the end.
        if end >= text_length:
            break

        # Preserve overlap.
        start = max(0, end - chunk_overlap)

        # Move to the next word boundary.
        while start < text_length and text[start] != " ":
            start += 1

        # Skip spaces.
        while start < text_length and text[start] == " ":
            start += 1

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    Split documents into smaller Document chunks.
    """

    chunked_documents = []

    for document in documents:

        text_chunks = chunk_text(
            text=document.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for index, chunk in enumerate(text_chunks):

            chunk_document = Document(
                id=f"{document.id}_chunk_{index}",
                text=chunk,
                source=document.source,
                metadata={
                    **document.metadata,
                    "parent_document_id": document.id,
                    "chunk_index": index,
                },
            )

            chunked_documents.append(chunk_document)

    return chunked_documents