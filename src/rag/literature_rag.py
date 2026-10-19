"""Literature RAG: answer "how does a drug work" questions from PubMed abstracts.

question -> search paper index -> top 5 abstracts -> qwen answers -> check [PMID ...] citations
"""
import re
import time

LITERATURE_SYSTEM_PROMPT = """You are a clinical research assistant. Answer using ONLY the research papers provided.
Rules:
1. Use only information from the papers below. Do not use outside knowledge.
2. After every claim, cite the paper like [PMID 12345678].
3. Never invent PMIDs. Only cite PMIDs that appear in the papers below.
4. If the papers do not answer the question, say: "The retrieved papers do not contain enough information to answer this question."
5. Do not give personal medical advice. For treatment decisions, tell the user to consult their doctor."""

MAX_ABSTRACT_CHARS = 1500  # keeps 5 papers well inside qwen's reading limit
PMID_PATTERN = re.compile(r"PMID[:\s]*(\d{6,9})", re.IGNORECASE)


def extract_pmids(text):
    """All PMIDs cited in the answer, in order, without duplicates."""
    return list(dict.fromkeys(PMID_PATTERN.findall(text)))


def build_literature_prompt(question, papers):
    blocks = []
    for p in papers:
        abstract = p["abstract"][:MAX_ABSTRACT_CHARS]
        blocks.append(f"[PMID {p['pmid']}] ({p.get('year', '')}, {p.get('journal', '')})\n"
                      f"Title: {p['title']}\nAbstract: {abstract}")
    sources = "\n\n".join(blocks)
    return f"PAPERS:\n\n{sources}\n\nQUESTION: {question}\n\nANSWER (cite [PMID ...] after every claim):"


def answer_literature_question(question, pubmed_retriever, llm=None, top_k=5):
    """Search papers, ask the LLM, and check that every cited PMID was really retrieved."""
    if llm is None:
        from src.llm.ollama_client import generate_response
        llm = generate_response

    start = time.time()
    papers = pubmed_retriever.search(question, top_k=top_k)
    retrieval_seconds = round(time.time() - start, 2)

    start = time.time()
    answer = llm(build_literature_prompt(question, papers), LITERATURE_SYSTEM_PROMPT)
    generation_seconds = round(time.time() - start, 2)

    retrieved = [p["pmid"] for p in papers]
    cited = extract_pmids(answer)
    valid = [pmid for pmid in cited if pmid in retrieved]
    invented = [pmid for pmid in cited if pmid not in retrieved]

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunk_ids": [],
        "retrieved_nct_ids": [],
        "cited_nct_ids": [],
        "valid_citations": [f"PMID {p}" for p in valid],
        "invented_citations": [f"PMID {p}" for p in invented],  # counted as "unsupported" in compare
        "retrieved_pmids": retrieved,
        "cited_pmids": cited,
        "retrieval_seconds": retrieval_seconds,
        "generation_seconds": generation_seconds,
        "filters": {},
    }