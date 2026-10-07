from unittest.mock import Mock

import pytest

from src.ingestion.clinicaltrials_fetcher import (
    API_URL,
    download_trials,
    parse_clinical_trials,
    search_clinical_trials,
)


def _study(nct_id: str = "NCT01234567") -> dict:
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": nct_id,
                "briefTitle": "Study of a new treatment",
            },
            "statusModule": {"overallStatus": "RECRUITING"},
            "conditionsModule": {"conditions": ["Condition A"]},
            "armsInterventionsModule": {
                "interventions": [{"name": "Treatment X"}]
            },
            "designModule": {"phases": ["PHASE2"]},
        }
    }


def test_parse_clinical_trials_returns_cited_trial_document() -> None:
    documents = parse_clinical_trials({"studies": [_study()]})

    assert len(documents) == 1
    document = documents[0]
    assert document.id == "NCT01234567"
    assert document.source == "ClinicalTrials.gov"
    assert "Status: RECRUITING" in document.text
    assert "Conditions: Condition A" in document.text
    assert document.metadata["interventions"] == ["Treatment X"]
    assert document.metadata["url"] == (
        "https://clinicaltrials.gov/study/NCT01234567"
    )


def test_search_clinical_trials_follows_next_page_token(monkeypatch) -> None:
    first_response = Mock()
    first_response.json.return_value = {
        "studies": [_study("NCT00000001")],
        "nextPageToken": "next",
    }
    second_response = Mock()
    second_response.json.return_value = {"studies": [_study("NCT00000002")]}
    get = Mock(side_effect=[first_response, second_response])
    monkeypatch.setattr(
        "src.ingestion.clinicaltrials_fetcher.requests.get",
        get,
    )

    documents = search_clinical_trials("condition A", max_results=2)

    assert [document.id for document in documents] == [
        "NCT00000001",
        "NCT00000002",
    ]
    assert get.call_args_list[0].args == (API_URL,)
    assert get.call_args_list[0].kwargs["params"]["query.term"] == "condition A"
    assert get.call_args_list[1].kwargs["params"]["pageToken"] == "next"
    assert all(call.kwargs["timeout"] == 30 for call in get.call_args_list)


@pytest.mark.parametrize(
    ("query", "max_results"),
    [("", 1), ("   ", 1), ("condition A", 0)],
)
def test_search_clinical_trials_rejects_invalid_arguments(
    query: str,
    max_results: int,
) -> None:
    with pytest.raises(ValueError):
        search_clinical_trials(query, max_results)


def test_parse_clinical_trials_includes_eligibility_outcomes_and_countries() -> None:
    study = _study()
    protocol = study["protocolSection"]
    protocol["eligibilityModule"] = {"eligibilityCriteria": "Inclusion: age 18+"}
    protocol["outcomesModule"] = {"primaryOutcomes": [{"measure": "Overall survival"}]}
    protocol["contactsLocationsModule"] = {
        "locations": [{"country": "India"}, {"country": "India"}, {"country": "Japan"}]
    }

    document = parse_clinical_trials({"studies": [study]})[0]

    assert "Eligibility: Inclusion: age 18+" in document.text
    assert "Primary outcomes: Overall survival" in document.text
    assert document.metadata["countries"] == ["India", "Japan"]


def test_parse_clinical_trials_skips_study_without_nct_id() -> None:
    study = _study()
    del study["protocolSection"]["identificationModule"]["nctId"]

    assert parse_clinical_trials({"studies": [study]}) == []


def test_download_trials_removes_duplicates_and_saves(monkeypatch, tmp_path) -> None:
    def fake_search(condition, max_results, search_field):
        assert search_field == "query.cond"
        return parse_clinical_trials(
            {"studies": [_study("NCT00000001"), _study("NCT00000002")]}
        )

    monkeypatch.setattr(
        "src.ingestion.clinicaltrials_fetcher.search_clinical_trials",
        fake_search,
    )
    output = tmp_path / "trials.json"

    documents = download_trials(
        conditions=["condition A", "condition B"],
        output_path=output,
        pause_seconds=0,
    )

    assert [document.id for document in documents] == ["NCT00000001", "NCT00000002"]
    assert output.exists()