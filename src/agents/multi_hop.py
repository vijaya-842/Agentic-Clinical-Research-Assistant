"""Multi-hop: questions that need TRIALS and PAPERS in one answer.

Example: "Which recruiting melanoma trials test pembrolizumab, and how does pembrolizumab work?"

  Hop 1  trial search        -> matching trials (LLM answer + code-written details)
  Hop 2  read the trials     -> which drugs they test (code, from ClinicalTrials.gov data)
  Hop 3  paper search x drug -> how each drug works (LLM answer with [PMID] citations)
  Aggregate                  -> one answer, one merged list of citations
"""
import re
from collections import Counter

DETAILS_MARKER = "\n\nTrial details (from ClinicalTrials.gov data):"
MAX_DRUGS = 2  # each drug costs one LLM call, so keep it small on CPU

TRIAL_WORDS = re.compile(r"\btrials?\b|\bstudies\b|\bstudy\b|\brecruiting\b|\benrolling\b", re.IGNORECASE)
MECHANISM_WORDS = re.compile(
    r"\bmechanisms?\b|\bhow\b.+\bworks?\b|\bexplain\b.+\b(drugs?|works?|mechanisms?)\b|\bwhat\b.+\btargets?\b",
    re.IGNORECASE,
)
SECOND_PART = re.compile(r",?\s*\b(and|then)\b\s+(explain|how|what|describe)\b", re.IGNORECASE)
NOT_A_DRUG = re.compile(
    r"placebo|surgery|surgical|radiation|radiotherapy|chemotherapy|standard of care|best supportive|"
    r"observation|questionnaire|biopsy|imaging|scan|blood|sample|procedure|behavioral|device|other",
    re.IGNORECASE,
)


def is_multi_hop(question):
    """True if the question asks about trials AND how drugs work."""
    return bool(TRIAL_WORDS.search(question) and MECHANISM_WORDS.search(question))


def trial_part(question):
    """'Find X trials, and explain how Y works' -> 'Find X trials'."""
    first = SECOND_PART.split(question)[0].strip()
    return first if len(first.split()) >= 3 else question


def clean_drug(name):
    """'Drug: Osimertinib 80 mg' -> 'Osimertinib'."""
    name = re.sub(r"^\s*(drug|biological|combination product)\s*:\s*", "", name, flags=re.IGNORECASE)
    name = re.split(r"\s+\d|\(|,|;", name)[0]
    return name.strip()


def extract_drugs(nct_ids, all_trials, max_drugs=MAX_DRUGS):
    """The most common drug names across the given trials (code, no LLM)."""
    by_id = {t["nct_id"]: t for t in all_trials}
    counts = Counter()
    for nct in nct_ids:
        seen = set()
        for intervention in (by_id.get(nct) or {}).get("interventions") or []:
            drug = clean_drug(str(intervention))
            if len(drug) < 3 or NOT_A_DRUG.search(drug) or drug.lower() in seen:
                continue
            seen.add(drug.lower())
            counts[drug.title() if drug.islower() else drug] += 1
    return [drug for drug, _ in counts.most_common(max_drugs)]


def answer_multi_hop(question, trial_tool, paper_tool, all_trials, max_drugs=MAX_DRUGS):
    """trial_tool(q) and paper_tool(q) are the existing tools; returns one combined result."""
    trials = trial_tool(trial_part(question))
    trial_text, _, details = trials["answer"].partition(DETAILS_MARKER)
    trial_ids = [c for c in trials.get("valid_citations", []) if c.startswith("NCT")]

    drugs = extract_drugs(trial_ids, all_trials, max_drugs)
    hops = [{"step": "trials", "query": trial_part(question), "trials": trial_ids}]
    papers = []
    for drug in drugs:
        paper = paper_tool(f"What is the mechanism of action of {drug}?")
        papers.append((drug, paper))
        hops.append({"step": "papers", "drug": drug, "pmids": paper.get("cited_pmids", [])})

    parts = ["Trials:\n" + trial_text.strip()]
    if papers:
        parts.append("How the drugs work:\n" + "\n\n".join(f"{drug}: {p['answer'].strip()}" for drug, p in papers))
    else:
        parts.append("How the drugs work:\nNo drug interventions were found in the matching trials to explain.")
    answer = "\n\n".join(parts)
    if details:
        answer += DETAILS_MARKER + details

    def merged(key):
        items = list(trials.get(key, []))
        for _, p in papers:
            items += [x for x in p.get(key, []) if x not in items]
        return items

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunk_ids": trials.get("retrieved_chunk_ids", []),
        "retrieved_nct_ids": trials.get("retrieved_nct_ids", []),
        "cited_nct_ids": trials.get("cited_nct_ids", []),
        "valid_citations": merged("valid_citations"),
        "invented_citations": merged("invented_citations"),
        "retrieved_pmids": merged("retrieved_pmids"),
        "cited_pmids": merged("cited_pmids"),
        "retrieval_seconds": round(trials.get("retrieval_seconds", 0) + sum(p.get("retrieval_seconds", 0) for _, p in papers), 2),
        "generation_seconds": round(trials.get("generation_seconds", 0) + sum(p.get("generation_seconds", 0) for _, p in papers), 2),
        "filters": trials.get("filters", {}),
        "drugs": drugs,
        "hops": hops,
    }