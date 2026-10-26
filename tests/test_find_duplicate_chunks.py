from scripts.find_duplicate_chunks import find_duplicates

H1 = "Trial NCT00000001: A\nSection: eligibility\n"
H2 = "Trial NCT00000002: B\nSection: eligibility\n"

CHUNKS = [
    {"chunk_id": "NCT00000001_eligibility_00", "text": H1 + "Age 18 or older."},
    {"chunk_id": "NCT00000001_eligibility_01", "text": H1 + "Age 18  or OLDER."},   # exact duplicate
    {"chunk_id": "NCT00000002_eligibility_00", "text": H2 + "Age 18 or older."},   # shared with trial 1
    {"chunk_id": "NCT00000002_summary_00", "text": H2 + "Something unique."},
]


def test_exact_duplicates_found():
    exact, _ = find_duplicates(CHUNKS)
    assert exact == [["NCT00000001_eligibility_00", "NCT00000001_eligibility_01"]]


def test_shared_text_across_trials_found():
    _, shared = find_duplicates(CHUNKS)
    assert len(shared) == 1
    assert "NCT00000002_eligibility_00" in shared[0]