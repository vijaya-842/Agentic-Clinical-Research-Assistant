"""
Turn downloaded clinical trials into section-level chunks.

Each trial becomes up to three sections, and each section is chunked:
    summary      - title, conditions, interventions, phase, status, summary
    eligibility  - inclusion / exclusion criteria
    outcomes     - primary and secondary outcome measures

Run from the project root:
    python -m src.ingestion.trial_chunks
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.ingestion.chunker import chunk_text
from src.ingestion.document import Document
from src.ingestion.loader import load_json_documents


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_TRIALS = PROJECT_ROOT / "data" / "raw" / "clinical_trials.json"
OUTPUT_CHUNKS = PROJECT_ROOT / "data" / "processed" / "trial_chunks.jsonl"
OUTPUT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "trial_chunks_manifest.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval.yaml"

# Small metadata copied onto every chunk (used later for filters and citations).
# Long fields such as the full eligibility text are NOT copied.
CHUNK_METADATA_FIELDS = [
    "nct_id",
    "title",
    "status",
    "phases",
    "conditions",
    "interventions",
    "sponsor",
    "countries",
    "url",
]


def _join(values: list[str], separator: str = ", ") -> str:
    return separator.join(value for value in values if value)


def trial_to_sections(trial: Document) -> dict[str, str]:
    """Split one trial into named sections. Empty sections are left out."""
    meta = trial.metadata

    summary_lines = [
        meta.get("title", ""),
        f"Conditions: {_join(meta.get('conditions', []))}",
        f"Interventions: {_join(meta.get('interventions', []))}",
        f"Phase: {_join(meta.get('phases', []))}",
        f"Status: {meta.get('status', '')}",
        f"Sponsor: {meta.get('sponsor', '')}",
        meta.get("summary", ""),
    ]

    outcomes_lines = []
    if meta.get("primary_outcomes"):
        outcomes_lines.append(
            f"Primary outcomes: {_join(meta['primary_outcomes'], '; ')}"
        )
    if meta.get("secondary_outcomes"):
        outcomes_lines.append(
            f"Secondary outcomes: {_join(meta['secondary_outcomes'], '; ')}"
        )

    sections = {
        "summary": "\n".join(line for line in summary_lines if line.strip()),
        "eligibility": meta.get("eligibility", ""),
        "outcomes": "\n".join(outcomes_lines),
    }
    return {name: text for name, text in sections.items() if text.strip()}


def trial_to_chunks(
    trial: Document,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Create section-level chunks for one trial."""
    meta = trial.metadata
    nct_id = meta.get("nct_id", trial.id)
    title = meta.get("title", "")
    base_metadata = {field: meta.get(field) for field in CHUNK_METADATA_FIELDS}

    chunks = []
    for section, text in trial_to_sections(trial).items():
        pieces = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for index, piece in enumerate(pieces):
            # A short header makes every chunk understandable on its own,
            # e.g. an eligibility chunk still says which trial it belongs to.
            header = f"Trial {nct_id}: {title}\nSection: {section}\n"

            chunks.append(
                Document(
                    id=f"{nct_id}_{section}_{index:02d}",
                    text=header + piece,
                    source=trial.source,
                    metadata={
                        **base_metadata,
                        "section": section,
                        "chunk_index": index,
                        "parent_document_id": trial.id,
                    },
                )
            )
    return chunks


def save_jsonl(documents: list[Document], output_path: Path) -> None:
    """Save one JSON object per line (easier to stream than one huge list)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        for document in documents:
            record = {
                "id": document.id,
                "text": document.text,
                "source": document.source,
                "metadata": document.metadata,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    input_path: Path = RAW_TRIALS,
    output_path: Path = OUTPUT_CHUNKS,
    manifest_path: Path = OUTPUT_MANIFEST,
) -> list[Document]:
    config = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    chunk_size = config["chunking"]["chunk_size"]
    chunk_overlap = config["chunking"]["chunk_overlap"]

    trials = load_json_documents(input_path)
    print(f"Loaded {len(trials)} trials from {input_path}")
    print(f"Chunk size: {chunk_size}, overlap: {chunk_overlap}")

    chunks: list[Document] = []
    for trial in trials:
        chunks.extend(trial_to_chunks(trial, chunk_size, chunk_overlap))

    save_jsonl(chunks, output_path)

    section_counts: dict[str, int] = {}
    for chunk in chunks:
        section = chunk.metadata["section"]
        section_counts[section] = section_counts.get(section, 0) + 1

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path.relative_to(PROJECT_ROOT)),
        "input_sha256": file_sha256(input_path),
        "trials": len(trials),
        "chunks": len(chunks),
        "chunks_per_section": section_counts,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "output_file": str(output_path.relative_to(PROJECT_ROOT)),
        "output_sha256": file_sha256(output_path),
    }
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print(f"Created {len(chunks)} chunks: {section_counts}")
    print(f"Saved chunks to {output_path}")
    print(f"Saved manifest to {manifest_path}")
    return chunks


if __name__ == "__main__":
    run()