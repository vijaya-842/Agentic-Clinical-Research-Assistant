"""Counting tool: answers "how many trials..." questions by counting ALL trials in the data.

Retrieval only shows the model 5 trials, so it cannot count.
This tool reads every trial in data/raw/clinical_trials.json and counts with plain Python.
"""
import json
import re
from pathlib import Path

RAW_TRIALS = Path("data/raw/clinical_trials.json")

# Words that mean "this is a counting question"
COUNT_PATTERN = re.compile(r"\b(how many|number of|count)\b", re.IGNORECASE)

# Status words in the question -> status value in the data.
# Longer phrases come first, so "not yet recruiting" is not read as "recruiting".
STATUS_WORDS = [
    ("not yet recruiting", "NOT_YET_RECRUITING"),
    ("active, not recruiting", "ACTIVE_NOT_RECRUITING"),
    ("active not recruiting", "ACTIVE_NOT_RECRUITING"),
    ("recruiting", "RECRUITING"),
    ("completed", "COMPLETED"),
    ("terminated", "TERMINATED"),
    ("withdrawn", "WITHDRAWN"),
    ("suspended", "SUSPENDED"),
]

# Cancer word in the question -> text to look for in the trial's conditions
CANCER_WORDS = {
    "pancrea": "pancrea",
    "breast": "breast",
    "lung": "lung",
    "colorectal": "colorectal",
    "colon": "colon",
    "rectal": "rect",
    "prostate": "prostat",
    "ovarian": "ovar",
    "melanoma": "melanoma",
    "leukemia": "leukemia",
    "lymphoma": "lymphoma",
    "glioblastoma": "glioblastoma",
    "liver": "liver",
    "hepatocellular": "hepatocellular",
    "gastric": "gastric",
    "stomach": "stomach",
    "bladder": "bladder",
    "kidney": "kidney",
    "renal": "renal",
    "cervical": "cervical",
    "head and neck": "head and neck",
}

ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4"}


def load_trials(path=RAW_TRIALS):
    """Read every trial from the raw JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Works whether each item is flat or keeps its fields under "metadata"
    return [item.get("metadata", item) for item in data]


def is_count_question(question):
    """True if the question asks for a number."""
    return bool(COUNT_PATTERN.search(question))


def parse_count_filters(question):
    """Find status, phase and cancer in the question."""
    q = question.lower()
    filters = {"status": None, "phase": None, "cancer": None}

    for phrase, value in STATUS_WORDS:
        if phrase in q:
            filters["status"] = value
            break

    match = re.search(r"phase\s*(iv|iii|ii|i|[1-4])\b", q)
    if match:
        number = ROMAN.get(match.group(1), match.group(1))
        filters["phase"] = "PHASE" + number

    for word, stem in CANCER_WORDS.items():
        if word in q:
            filters["cancer"] = stem
            break

    return filters


def trial_matches(trial, filters):
    """Check one trial against the filters."""
    if filters["status"]:
        status = str(trial.get("status", "")).upper().replace(" ", "_").replace(",", "")
        if status != filters["status"]:
            return False

    if filters["phase"]:
        phases = [str(p).upper().replace(" ", "") for p in trial.get("phases") or []]
        if filters["phase"] not in phases:
            return False

    if filters["cancer"]:
        conditions = " ".join(trial.get("conditions") or []).lower()
        if filters["cancer"] not in conditions:
            return False

    return True


def count_trials(trials, filters):
    """Return how many trials match, and their NCT IDs."""
    matching = [t["nct_id"] for t in trials if trial_matches(t, filters)]
    return len(matching), matching


def describe(filters):
    """Turn filters into words, e.g. 'RECRUITING, PHASE3, pancrea'."""
    parts = [v for v in filters.values() if v]
    return ", ".join(parts) if parts else "no filters"


def answer_count_question(question, trials):
    """Answer a counting question with code, not with the LLM."""
    filters = parse_count_filters(question)
    count, ids = count_trials(trials, filters)
    examples = ids[:5]
    answer = (
        f"There are {count} matching trials in our dataset of {len(trials)} oncology trials "
        f"(filters: {describe(filters)})."
    )
    if examples:
        answer += " Examples: " + ", ".join(f"[{i}]" for i in examples) + "."
    return {"answer": answer, "count": count, "filters": filters, "example_ids": examples}


if __name__ == "__main__":
    all_trials = load_trials()
    q = "How many recruiting pancreatic cancer trials are in the dataset?"
    print("Question:", q)
    print("Counting question?", is_count_question(q))
    print(answer_count_question(q, all_trials)["answer"])