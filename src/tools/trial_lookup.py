"""Trial lookup tool: questions that name a trial ID (NCT...) get that trial's facts directly.

Dense search is weak at matching ID strings, so we skip search and read the record.
The answer is written by code from the raw data, so every fact is exact.
"""
import re

NCT_ID_PATTERN = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)


def find_nct_ids(question):
    """All NCT IDs in the question, upper-case, in order, without duplicates."""
    return list(dict.fromkeys(m.upper() for m in NCT_ID_PATTERN.findall(question)))


def as_text_list(values):
    """Outcomes can be strings or dicts like {'measure': ...}; return plain strings."""
    items = []
    for v in values or []:
        items.append(v.get("measure", str(v)) if isinstance(v, dict) else str(v))
    return items


def describe_trial(trial):
    phases = ", ".join(trial.get("phases") or []) or "N/A"
    outcomes = "; ".join(as_text_list(trial.get("primary_outcomes"))[:3]) or "not listed"
    lines = [
        f"[{trial['nct_id']}] {trial.get('title', '')}",
        f"- Sponsor: {trial.get('sponsor', 'not listed')}",
        f"- Status: {trial.get('status', 'unknown')} | Phase: {phases}",
        f"- Conditions: {', '.join(trial.get('conditions') or []) or 'not listed'}",
        f"- Interventions: {', '.join(trial.get('interventions') or []) or 'not listed'}",
        f"- Primary outcome: {outcomes}",
    ]
    summary = (trial.get("summary") or "").strip()
    if summary:
        lines.append(f"- Summary: {summary[:400]}{'...' if len(summary) > 400 else ''}")
    return "\n".join(lines)


def answer_trial_lookup(question, all_trials):
    """Return a result dict (same shape as the other tools) for questions naming trial IDs."""
    by_id = {t["nct_id"]: t for t in all_trials}
    asked = find_nct_ids(question)
    found = [nct for nct in asked if nct in by_id]
    missing = [nct for nct in asked if nct not in by_id]

    parts = [describe_trial(by_id[nct]) for nct in found]
    if missing:
        parts.append("Not in our dataset: " + ", ".join(missing))
    return {
        "question": question,
        "answer": "\n\n".join(parts) if parts else "No trial ID found in the question.",
        "retrieved_chunk_ids": [],
        "retrieved_nct_ids": found,
        "cited_nct_ids": found,
        "valid_citations": found,
        "invented_citations": [],
        "retrieval_seconds": 0.0,
        "generation_seconds": 0.0,
        "filters": {},
    }