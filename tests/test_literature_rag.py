from src.rag.literature_rag import (
    answer_literature_question,
    build_literature_prompt,
    extract_pmids,
)

PAPERS = [
    {"pmid": "11111111", "title": "Osimertinib review", "abstract": "Irreversible EGFR inhibitor.",
     "year": "2020", "journal": "J Onc"},
    {"pmid": "22222222", "title": "T790M resistance", "abstract": "Targets T790M mutation.",
     "year": "2019", "journal": "Lung"},
]


class FakeRetriever:
    def search(self, question, top_k=5):
        return PAPERS[:top_k]


def test_extract_pmids():
    text = "It blocks EGFR [PMID 11111111]. Also T790M [PMID: 22222222] and again [PMID 11111111]."
    assert extract_pmids(text) == ["11111111", "22222222"]


def test_prompt_contains_papers_and_question():
    prompt = build_literature_prompt("How does osimertinib work?", PAPERS)
    assert "[PMID 11111111]" in prompt
    assert "Irreversible EGFR inhibitor." in prompt
    assert "QUESTION: How does osimertinib work?" in prompt


def test_valid_and_invented_citations():
    def fake_llm(prompt, system_prompt):
        return "It inhibits EGFR [PMID 11111111] and cures everything [PMID 99999999]."

    result = answer_literature_question("How does osimertinib work?", FakeRetriever(), llm=fake_llm)
    assert result["retrieved_pmids"] == ["11111111", "22222222"]
    assert result["valid_citations"] == ["PMID 11111111"]
    assert result["invented_citations"] == ["PMID 99999999"]