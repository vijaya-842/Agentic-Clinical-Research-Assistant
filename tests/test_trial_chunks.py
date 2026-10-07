import json

from src.ingestion.document import Document
from src.ingestion.trial_chunks import save_jsonl, trial_to_chunks, trial_to_sections


def _trial(**overrides) -> Document:
    metadata = {
        "nct_id": "NCT01234567",
        "title": "Study of Drug X in Lung Cancer",
        "status": "RECRUITING",
        "phases": ["PHASE3"],
        "conditions": ["Non-small Cell Lung Cancer"],
        "interventions": ["Drug X"],
        "sponsor": "Example Sponsor",
        "countries": ["India"],
        "summary": "This study tests Drug X.",
        "eligibility": "Inclusion: age 18 or older. Exclusion: brain metastases.",
        "primary_outcomes": ["Overall survival"],
        "secondary_outcomes": ["Response rate"],
        "url": "https://clinicaltrials.gov/study/NCT01234567",
    }
    metadata.update(overrides)
    return Document(
        id="NCT01234567",
        text="full text",
        source="ClinicalTrials.gov",
        metadata=metadata,
    )


def test_trial_to_sections_creates_three_sections() -> None:
    sections = trial_to_sections(_trial())

    assert list(sections) == ["summary", "eligibility", "outcomes"]
    assert "Drug X" in sections["summary"]
    assert "brain metastases" in sections["eligibility"]
    assert "Primary outcomes: Overall survival" in sections["outcomes"]


def test_trial_to_sections_skips_empty_sections() -> None:
    sections = trial_to_sections(
        _trial(eligibility="", primary_outcomes=[], secondary_outcomes=[])
    )

    assert list(sections) == ["summary"]


def test_trial_to_chunks_ids_header_and_metadata() -> None:
    chunks = trial_to_chunks(_trial(), chunk_size=500, chunk_overlap=100)

    ids = [chunk.id for chunk in chunks]
    assert ids == [
        "NCT01234567_summary_00",
        "NCT01234567_eligibility_00",
        "NCT01234567_outcomes_00",
    ]

    eligibility = chunks[1]
    assert eligibility.text.startswith(
        "Trial NCT01234567: Study of Drug X in Lung Cancer\nSection: eligibility\n"
    )
    assert eligibility.metadata["section"] == "eligibility"
    assert eligibility.metadata["phases"] == ["PHASE3"]
    assert eligibility.metadata["parent_document_id"] == "NCT01234567"
    # Long fields are not copied onto every chunk.
    assert "eligibility" not in eligibility.metadata
    assert "summary" not in eligibility.metadata


def test_long_eligibility_is_split_into_numbered_chunks() -> None:
    long_text = " ".join(f"criterion{i}" for i in range(300))
    chunks = trial_to_chunks(
        _trial(eligibility=long_text), chunk_size=500, chunk_overlap=100
    )

    eligibility_ids = [c.id for c in chunks if c.metadata["section"] == "eligibility"]
    assert len(eligibility_ids) > 1
    assert eligibility_ids[0] == "NCT01234567_eligibility_00"
    assert eligibility_ids[1] == "NCT01234567_eligibility_01"


def test_save_jsonl_writes_one_record_per_line(tmp_path) -> None:
    chunks = trial_to_chunks(_trial(), chunk_size=500, chunk_overlap=100)
    output = tmp_path / "chunks.jsonl"

    save_jsonl(chunks, output)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(chunks)
    assert json.loads(lines[0])["id"] == "NCT01234567_summary_00"