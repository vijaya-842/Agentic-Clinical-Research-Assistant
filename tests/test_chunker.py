from src.ingestion.document import Document
from src.ingestion.chunker import chunk_documents


sample_text = """
Metformin is commonly used as a treatment for Type 2 diabetes.
It helps reduce blood glucose levels and improve insulin sensitivity.
Common side effects may include nausea, diarrhea, and abdominal discomfort.
Lifestyle modifications such as diet and physical activity are also important.
"""


document = Document(
    id="test_001",
    text=sample_text,
    source="test",
    metadata={
        "title": "Sample Diabetes Document"
    },
)


chunks = chunk_documents(
    documents=[document],
    chunk_size=100,
    chunk_overlap=20,
)


print(f"\nOriginal document length: {len(sample_text)} characters")
print(f"Number of chunks: {len(chunks)}\n")


for chunk in chunks:

    print("=" * 60)
    print(f"Chunk ID: {chunk.id}")
    print(f"Parent ID: {chunk.metadata['parent_document_id']}")
    print(f"Chunk Index: {chunk.metadata['chunk_index']}")
    print("\nText:")
    print(chunk.text)
    print()