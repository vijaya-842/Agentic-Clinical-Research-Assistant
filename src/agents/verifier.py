"""Critic / Verifier: checks an answer BEFORE it is shown to the user.

Checks (all done by code, no LLM, so they are fast and always the same):
  1. Invented citations  - IDs in the answer that were not in the retrieved sources
  2. Status claims       - the answer says a trial is "recruiting / open / enrolling",
                           but ClinicalTrials.gov data says it is not
  3. Citation coverage   - how many of the answer's sentences cite a source

If a problem is found, a short "Verifier note" is added to the answer so the
user sees the correction, and the result records what was checked.
"""
import re

CODE_TOOLS = {"count_trials", "trial_lookup", "small_talk", "safety_refusal"}  # written by code, not the LLM
DETAILS_MARKER = "\n\nTrial details (from ClinicalTrials.gov data):"

NCT_PATTERN = re.compile(r"NCT\d{8}")
PMID_PATTERN = re.compile(r"PMID[:\s]*\d{6,9}", re.IGNORECASE)
OPEN_CLAIM = re.compile(
    r"\b(recruiting|enrolling|open to (new )?(patients|participants|enrollment)"
    r"|accepting (new )?(patients|participants)|currently open)\b",
    re.IGNORECASE,
)
NEGATION = re.compile(r"\b(not|no longer|never|closed)\b", re.IGNORECASE)
MIN_COVERAGE = 0.5  # below this, too many sentences have no source
REFUSAL = re.compile(r"do not contain enough information|consult", re.IGNORECASE)


def llm_text(answer):
    """Only check the part the LLM wrote, not the trial details that code added."""
    return answer.split(DETAILS_MARKER)[0]


def sentences(text):
    """Split into sentences / lines, with underscores as spaces (ACTIVE_NOT_RECRUITING -> active not recruiting)."""
    text = text.replace("_", " ")
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def status_issues(text, by_id):
    """Find trials the answer calls recruiting/open while the data says otherwise."""
    issues = {}
    for sentence in sentences(text):
        if not OPEN_CLAIM.search(sentence) or NEGATION.search(sentence):
            continue
        for nct in NCT_PATTERN.findall(sentence):
            trial = by_id.get(nct)
            status = str(trial.get("status", "")).upper() if trial else ""
            if trial and status != "RECRUITING":
                issues[nct] = (f"{nct} is described as recruiting/open, but ClinicalTrials.gov "
                               f"lists its status as {status}.")
    return list(issues.values())


def citation_coverage(text):
    """Share of factual sentences (5+ words) that cite at least one trial or paper."""
    factual = [s for s in sentences(text) if len(s.split()) >= 5 and not REFUSAL.search(s)]
    if not factual:
        return None
    cited = [s for s in factual if NCT_PATTERN.search(s) or PMID_PATTERN.search(s)]
    return round(len(cited) / len(factual), 2)


def verify_answer(result, all_trials, tool):
    """Return the result with a 'verification' entry, and a note added to the answer if needed."""
    if tool in CODE_TOOLS:
        result["verification"] = {"status": "skipped", "reason": "answer written by code", "issues": []}
        return result

    by_id = {t["nct_id"]: t for t in all_trials}
    text = llm_text(result["answer"])
    issues = []
    for cid in result.get("invented_citations", []):
        issues.append(f"{cid} was cited but is not in the retrieved sources.")
    issues += status_issues(text, by_id)
    coverage = citation_coverage(text)
    if coverage is not None and coverage < MIN_COVERAGE:
        issues.append(f"Only {coverage:.0%} of the answer's sentences cite a source; "
                      "uncited statements may come from the model's memory, not the sources.")

    result["verification"] = {
        "status": "warning" if issues else "pass",
        "issues": issues,
        "citation_coverage": coverage,
    }
    if issues:
        result["answer"] += "\n\nVerifier note:\n" + "\n".join(f"- {i}" for i in issues)
    return result