"""Find duplicate chunks in data/processed/trial_chunks.jsonl (report only, changes nothing).

Two kinds of duplicates:
  1. EXACT  - the same chunk text (header included) appears more than once.
              These are true duplicates: same trial, same section, same words.
  2. SHARED - different trials have the same body text (header removed),
              e.g. a sponsor reuses the same eligibility wording in many trials.
              These are NOT errors: each copy belongs to a different trial.

Run from the project root:
    python -m scripts.find_duplicate_chunks
"""
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

CHUNKS_FILE = Path("data/processed/trial_chunks.jsonl")
TEXT_FIELDS = ("text", "chunk_text", "content")
ID_FIELDS = ("chunk_id", "id")
HEADER = re.compile(r"^Trial NCT\d{8}:.*?\nSection: .*?\n", re.DOTALL)


def field(record, names):
    for name in names:
        if name in record:
            return record[name]
    raise KeyError(f"None of {names} found. Keys are: {list(record)}")


def normalize(text):
    """Lowercase and squeeze spaces so tiny whitespace differences do not hide duplicates."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def fingerprint(text):
    return hashlib.md5(normalize(text).encode("utf-8")).hexdigest()


def load_chunks(path=CHUNKS_FILE):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_duplicates(chunks):
    exact = defaultdict(list)   # full text  -> chunk ids
    shared = defaultdict(list)  # body text  -> chunk ids
    for chunk in chunks:
        text, chunk_id = field(chunk, TEXT_FIELDS), field(chunk, ID_FIELDS)
        exact[fingerprint(text)].append(chunk_id)
        shared[fingerprint(HEADER.sub("", text))].append(chunk_id)
    exact_groups = [ids for ids in exact.values() if len(ids) > 1]
    shared_groups = [ids for ids in shared.values()
                     if len({cid.split("_")[0] for cid in ids}) > 1]  # more than one trial
    return exact_groups, shared_groups


def main():
    chunks = load_chunks()
    exact_groups, shared_groups = find_duplicates(chunks)
    extra_exact = sum(len(g) - 1 for g in exact_groups)
    extra_shared = sum(len(g) - 1 for g in shared_groups)

    print(f"Chunks: {len(chunks)}")
    print(f"\n1. EXACT duplicates: {len(exact_groups)} groups, {extra_exact} extra copies "
          f"({100 * extra_exact / len(chunks):.2f}% of chunks)")
    for ids in exact_groups[:5]:
        print("   ", ids[:4])
    print(f"\n2. SHARED text across different trials: {len(shared_groups)} groups, "
          f"{extra_shared} chunks ({100 * extra_shared / len(chunks):.2f}%)")
    for ids in sorted(shared_groups, key=len, reverse=True)[:5]:
        print(f"    {len(ids)} trials share one text, e.g. {ids[:3]}")


if __name__ == "__main__":
    main()