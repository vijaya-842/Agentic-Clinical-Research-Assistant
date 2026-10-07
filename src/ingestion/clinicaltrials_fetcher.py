"""
Fetch clinical trials from the ClinicalTrials.gov API (v2) and convert them
into project Document objects.

Download the oncology corpus (run from the project root):
    python -m src.ingestion.clinicaltrials_fetcher
"""

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import requests

from src.ingestion.document import Document


API_URL = "https://clinicaltrials.gov/api/v2/studies"
SOURCE = "ClinicalTrials.gov"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "clinical_trials.json"

# Oncology topics suggested in the handbook (section 7.2).
ONCOLOGY_CONDITIONS = [
    "breast cancer",
    "non-small cell lung cancer",
    "melanoma",
    "colorectal cancer",
    "prostate cancer",
    "leukemia",
    "lymphoma",
    "multiple myeloma",
    "pancreatic cancer",
    "ovarian cancer",
]


# ---------------------------------------------------------------------------
# Small helpers that never crash on missing or malformed fields
# ---------------------------------------------------------------------------

def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _module(protocol: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a protocol module, or an empty dict if it is missing or malformed."""
    module = protocol.get(name, {})
    return module if isinstance(module, dict) else {}


def _names(items: Any, key: str) -> list[str]:
    """Extract one text field from a list of dicts, e.g. intervention names."""
    if not isinstance(items, list):
        return []
    return [
        value
        for item in items
        if isinstance(item, dict)
        if (value := _text(item.get(key)))
    ]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _trial_to_document(study: dict[str, Any]) -> Document | None:
    protocol = study.get("protocolSection", {})
    if not isinstance(protocol, dict):
        return None

    identification = _module(protocol, "identificationModule")
    status_module = _module(protocol, "statusModule")
    conditions_module = _module(protocol, "conditionsModule")
    interventions_module = _module(protocol, "armsInterventionsModule")
    design_module = _module(protocol, "designModule")
    sponsor_module = _module(protocol, "sponsorCollaboratorsModule")
    description_module = _module(protocol, "descriptionModule")
    eligibility_module = _module(protocol, "eligibilityModule")
    outcomes_module = _module(protocol, "outcomesModule")
    locations_module = _module(protocol, "contactsLocationsModule")

    nct_id = _text(identification.get("nctId"))
    if not nct_id:
        return None

    title = _text(identification.get("briefTitle")) or "Untitled clinical trial"
    status = _text(status_module.get("overallStatus"))
    conditions = _strings(conditions_module.get("conditions"))
    interventions = _names(interventions_module.get("interventions"), "name")
    phases = _strings(design_module.get("phases"))
    study_type = _text(design_module.get("studyType"))

    enrollment_info = design_module.get("enrollmentInfo", {})
    enrollment = (
        enrollment_info.get("count") if isinstance(enrollment_info, dict) else None
    )

    lead_sponsor = sponsor_module.get("leadSponsor", {})
    sponsor = _text(lead_sponsor.get("name")) if isinstance(lead_sponsor, dict) else ""

    summary = _text(description_module.get("briefSummary"))
    eligibility = _text(eligibility_module.get("eligibilityCriteria"))
    primary_outcomes = _names(outcomes_module.get("primaryOutcomes"), "measure")
    secondary_outcomes = _names(outcomes_module.get("secondaryOutcomes"), "measure")
    countries = sorted(set(_names(locations_module.get("locations"), "country")))

    lines = [title, f"Trial ID: {nct_id}"]
    if status:
        lines.append(f"Status: {status}")
    if phases:
        lines.append(f"Phases: {', '.join(phases)}")
    if study_type:
        lines.append(f"Study type: {study_type}")
    if conditions:
        lines.append(f"Conditions: {', '.join(conditions)}")
    if interventions:
        lines.append(f"Interventions: {', '.join(interventions)}")
    if sponsor:
        lines.append(f"Sponsor: {sponsor}")
    if enrollment is not None:
        lines.append(f"Enrollment: {enrollment}")
    if primary_outcomes:
        lines.append(f"Primary outcomes: {'; '.join(primary_outcomes)}")
    if secondary_outcomes:
        lines.append(f"Secondary outcomes: {'; '.join(secondary_outcomes)}")
    if summary:
        lines.append(f"Summary: {summary}")
    if eligibility:
        lines.append(f"Eligibility: {eligibility}")

    return Document(
        id=nct_id,
        text="\n".join(lines),
        source=SOURCE,
        metadata={
            "nct_id": nct_id,
            "title": title,
            "status": status,
            "phases": phases,
            "study_type": study_type,
            "conditions": conditions,
            "interventions": interventions,
            "sponsor": sponsor,
            "enrollment": enrollment,
            "countries": countries,
            "primary_outcomes": primary_outcomes,
            "secondary_outcomes": secondary_outcomes,
            # Kept separately so we can build section-level chunks later.
            "summary": summary,
            "eligibility": eligibility,
            "url": f"https://clinicaltrials.gov/study/{nct_id}",
        },
    )


def parse_clinical_trials(data: dict[str, Any]) -> list[Document]:
    """Convert a ClinicalTrials.gov API response into normalized documents."""
    studies = data.get("studies", [])
    if not isinstance(studies, list):
        raise ValueError("ClinicalTrials.gov response field 'studies' must be a list.")

    documents = []
    for study in studies:
        if isinstance(study, dict):
            document = _trial_to_document(study)
            if document is not None:
                documents.append(document)
    return documents


# ---------------------------------------------------------------------------
# Searching the API
# ---------------------------------------------------------------------------

def search_clinical_trials(
    query: str,
    max_results: int = 20,
    search_field: str = "query.term",
) -> list[Document]:
    """
    Search ClinicalTrials.gov and return matching trials as documents.

    search_field: "query.term" searches all fields;
                  "query.cond" searches only the condition field.
    """
    if not query.strip():
        raise ValueError("query must not be empty.")
    if max_results < 1:
        raise ValueError("max_results must be at least 1.")

    documents: list[Document] = []
    page_token: str | None = None

    while len(documents) < max_results:
        params: dict[str, Any] = {
            search_field: query,
            "format": "json",
            "pageSize": min(max_results - len(documents), 1000),
        }
        if page_token:
            params["pageToken"] = page_token

        response = requests.get(API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        documents.extend(parse_clinical_trials(data))

        next_page_token = data.get("nextPageToken")
        if not isinstance(next_page_token, str) or not next_page_token:
            break
        page_token = next_page_token

    return documents[:max_results]


# ---------------------------------------------------------------------------
# Bulk download for the project corpus
# ---------------------------------------------------------------------------

def save_documents(documents: list[Document], output_path: Path) -> None:
    """Save documents as JSON in the format loader.py reads."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            [asdict(document) for document in documents],
            file,
            indent=2,
            ensure_ascii=False,
        )


def download_trials(
    conditions: list[str] = ONCOLOGY_CONDITIONS,
    max_per_condition: int = 500,
    output_path: Path = DEFAULT_OUTPUT,
    pause_seconds: float = 0.5,
) -> list[Document]:
    """Download trials for every condition, remove duplicates and save them."""
    trials: dict[str, Document] = {}

    for condition in conditions:
        print(f"Downloading: {condition}")
        try:
            documents = search_clinical_trials(
                condition,
                max_results=max_per_condition,
                search_field="query.cond",
            )
        except requests.RequestException as error:
            print(f"  Failed for '{condition}': {error}")
            continue

        # The same trial can appear under several conditions;
        # keying by NCT ID keeps one copy.
        for document in documents:
            trials[document.id] = document

        print(f"  Got {len(documents)} trials (unique so far: {len(trials)})")
        time.sleep(pause_seconds)

    unique_documents = list(trials.values())
    save_documents(unique_documents, output_path)
    print(f"\nSaved {len(unique_documents)} unique trials to {output_path}")
    return unique_documents


if __name__ == "__main__":
    download_trials(max_per_condition=500)