"""
Trial fact sheets.

Each retrieved chunk is only a small piece of a trial. To stop the LLM from
missing (or borrowing) key details, we add a short fact sheet for every
retrieved trial, taken directly from the downloaded data:
sponsor, status, phase, conditions, interventions and primary outcomes.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_TRIALS = PROJECT_ROOT / "data" / "raw" / "clinical_trials.json"

MAX_LIST_ITEMS = 5  # keep fact sheets short so the prompt stays small


def load_trial_facts(path: Path = RAW_TRIALS) -> dict[str, dict]:
    """Read the downloaded trials and keep the metadata, keyed by NCT ID."""
    with open(path, encoding="utf-8") as file:
        trials = json.load(file)
    return {trial["id"]: trial.get("metadata", {}) for trial in trials}


def _short_list(values, separator: str = ", ") -> str:
    values = [value for value in (values or []) if value]
    text = separator.join(values[:MAX_LIST_ITEMS])
    if len(values) > MAX_LIST_ITEMS:
        text += f"{separator}... ({len(values)} in total)"
    return text or "not listed"


def format_fact_sheet(nct_id: str, facts: dict) -> str:
    """One small block of verified facts about a trial."""
    return "\n".join([
        f"FACT SHEET for {nct_id}",
        f"Title: {facts.get('title') or 'not listed'}",
        f"Sponsor: {facts.get('sponsor') or 'not listed'}",
        f"Status: {facts.get('status') or 'not listed'}",
        f"Phase: {_short_list(facts.get('phases'))}",
        f"Conditions: {_short_list(facts.get('conditions'))}",
        f"Interventions: {_short_list(facts.get('interventions'))}",
        f"Primary outcomes: {_short_list(facts.get('primary_outcomes'), '; ')}",
    ])


def build_prompt_with_facts(question: str, results: list, trial_facts: dict) -> str:
    """Like Naive RAG's prompt, but each source starts with its trial's fact sheet."""
    sources = []
    for number, result in enumerate(results, start=1):
        nct_id = result.metadata.get("nct_id", "")
        section = result.metadata.get("section", "")
        facts = trial_facts.get(nct_id)
        fact_sheet = format_fact_sheet(nct_id, facts) if facts else f"(no fact sheet for {nct_id})"
        sources.append(
            f"[{number}] Trial {nct_id}\n{fact_sheet}\n"
            f"Best matching text ({section}):\n{result.text}"
        )

    return (
        "SOURCES:\n\n"
        + "\n\n".join(sources)
        + f"\n\nQUESTION: {question}\n\n"
        + "Answer using only the sources above and cite trial IDs like [NCT01234567]."
    )