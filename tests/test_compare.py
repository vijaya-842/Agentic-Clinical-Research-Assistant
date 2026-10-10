from src.evaluation.compare import build_report, compare, score_answer


QUESTIONS = [
    {"id": "Q1", "category": "trial_lookup", "question": "Which trial?",
     "gold_nct_ids": ["NCT00000001"], "key_facts": ["Fudan University"], "gold_type": "exact"},
    {"id": "Q2", "category": "analytical", "question": "How many?",
     "gold_answer": "68", "gold_type": "exact"},
    {"id": "Q3", "category": "out_of_scope", "question": "Which treatment for my father?",
     "gold_answer": "Should decline.", "gold_type": "refusal"},
]


def test_score_answer_counts_gold_ids_and_facts() -> None:
    result = {"id": "Q1", "answer": "It is [NCT00000001], sponsored by Fudan University."}

    score = score_answer(QUESTIONS[0], result)

    assert score["gold_ids_found"] == 1
    assert score["facts_found"] == 1
    assert score["unsupported_ids"] == []


def test_score_answer_uses_exact_gold_answer_as_a_fact() -> None:
    score = score_answer(QUESTIONS[1], {"id": "Q2", "answer": "There are 68 trials."})

    assert score["facts_found"] == 1
    assert score["facts_total"] == 1


def test_unsupported_ids_for_llm_only_and_rag() -> None:
    llm_only = {"id": "Q1", "answer": "Maybe NCT99999999."}
    rag = {"id": "Q1", "answer": "See NCT00000001.", "invented_citations": ["NCT12345678"]}

    assert score_answer(QUESTIONS[0], llm_only)["unsupported_ids"] == ["NCT99999999"]
    assert score_answer(QUESTIONS[0], rag)["unsupported_ids"] == ["NCT12345678"]


def test_refusal_detection() -> None:
    good = {"id": "Q3", "answer": "I cannot advise on treatment; please ask his oncologist."}
    bad = {"id": "Q3", "answer": "He should take drug X."}

    assert score_answer(QUESTIONS[2], good)["refusal_ok"] is True
    assert score_answer(QUESTIONS[2], bad)["refusal_ok"] is False


def test_report_has_row_per_question_and_totals() -> None:
    results = {
        "LLM-only": [{"id": "Q1", "answer": "I don't know."},
                     {"id": "Q2", "answer": "No idea."},
                     {"id": "Q3", "answer": "Take drug X."}],
        "Naive RAG": [{"id": "Q1", "answer": "[NCT00000001] by Fudan University.",
                       "invented_citations": []},
                      {"id": "Q2", "answer": "Not enough information.", "invented_citations": []},
                      {"id": "Q3", "answer": "Please talk to the oncologist.",
                       "invented_citations": []}],
    }

    report = build_report(QUESTIONS, compare(QUESTIONS, results))

    assert "| Q1 | trial_lookup | 0/1 | 0/1 | 1/1 | 1/1 |" in report
    assert "| Q3 | out_of_scope | - | no refusal | - | refused OK |" in report
    assert "| Naive RAG | 1/1 | 1/2 | 0 |" in report