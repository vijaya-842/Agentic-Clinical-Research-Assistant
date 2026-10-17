"""Download PubMed abstracts (title + abstract) for the literature corpus.

Uses NCBI E-utilities (free, no key needed):
  1. esearch -> find PubMed IDs (PMIDs) for a search term
  2. efetch  -> download title, abstract, journal, year for those PMIDs
Saves everything to data/raw/pubmed_papers.json
"""
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OUTPUT_FILE = Path("data/raw/pubmed_papers.json")
PAUSE_SECONDS = 0.4  # NCBI allows max 3 requests per second without an API key

# Same 10 cancers as the trial data -> review papers about each cancer
CANCERS = [
    "breast cancer", "lung cancer", "colorectal cancer", "prostate cancer",
    "pancreatic cancer", "melanoma", "leukemia", "lymphoma",
    "ovarian cancer", "glioblastoma",
]

# Common oncology drugs (many appear in our trials) -> papers about how they work
DRUGS = [
    "osimertinib", "pembrolizumab", "nivolumab", "atezolizumab", "durvalumab",
    "ipilimumab", "trastuzumab", "tucatinib", "capecitabine", "bevacizumab",
    "cetuximab", "olaparib", "palbociclib", "abemaciclib", "enzalutamide",
    "abiraterone", "imatinib", "ibrutinib", "venetoclax", "rituximab",
    "lenalidomide", "gemcitabine", "paclitaxel", "docetaxel", "temozolomide",
]

PAPERS_PER_CANCER = 100
PAPERS_PER_DRUG = 30


def search_pubmed(term, max_results=100):
    """Return a list of PMIDs (strings) for a search term."""
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    response = requests.get(f"{BASE_URL}/esearch.fcgi", params=params, timeout=30)
    response.raise_for_status()
    time.sleep(PAUSE_SECONDS)
    return response.json()["esearchresult"]["idlist"]


def element_text(element):
    """All text inside an XML element, including text inside <i>, <sup>, etc."""
    return "".join(element.itertext()).strip() if element is not None else ""


def parse_pubmed_xml(xml_text):
    """Turn efetch XML into a list of paper dicts."""
    root = ET.fromstring(xml_text)
    papers = []
    for article in root.findall(".//PubmedArticle"):
        pmid = element_text(article.find(".//MedlineCitation/PMID"))
        title = element_text(article.find(".//Article/ArticleTitle"))

        # An abstract can have several parts (BACKGROUND, METHODS, RESULTS ...)
        parts = []
        for part in article.findall(".//Article/Abstract/AbstractText"):
            label = part.get("Label")
            text = element_text(part)
            if text:
                parts.append(f"{label}: {text}" if label else text)
        abstract = " ".join(parts)

        journal = element_text(article.find(".//Article/Journal/Title"))
        year = element_text(article.find(".//Article/Journal/JournalIssue/PubDate/Year"))
        if not year:  # some papers only have <MedlineDate>2019 Jan-Feb</MedlineDate>
            year = element_text(article.find(".//Article/Journal/JournalIssue/PubDate/MedlineDate"))[:4]
        pub_types = [element_text(p) for p in article.findall(".//PublicationTypeList/PublicationType")]

        if pmid and title and abstract:
            papers.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": year,
                "publication_types": pub_types,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
    return papers


def fetch_abstracts(pmids, batch_size=100):
    """Download paper details for many PMIDs, 100 at a time."""
    papers = []
    for start in range(0, len(pmids), batch_size):
        batch = pmids[start:start + batch_size]
        params = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"}
        response = requests.get(f"{BASE_URL}/efetch.fcgi", params=params, timeout=60)
        response.raise_for_status()
        papers.extend(parse_pubmed_xml(response.text))
        time.sleep(PAUSE_SECONDS)
    return papers


def build_search_terms():
    """The list of (search term, how many papers) we ask PubMed for."""
    terms = []
    for cancer in CANCERS:
        terms.append((f'"{cancer}"[Title] AND review[pt] AND hasabstract', PAPERS_PER_CANCER))
    for drug in DRUGS:
        terms.append((f'{drug}[Title] AND (mechanism OR review[pt]) AND hasabstract', PAPERS_PER_DRUG))
    return terms


def download_papers(output_path=OUTPUT_FILE):
    all_pmids = []
    for term, max_results in build_search_terms():
        pmids = search_pubmed(term, max_results)
        print(f"{len(pmids):4d} papers  <- {term}")
        all_pmids.extend(pmids)

    unique_pmids = list(dict.fromkeys(all_pmids))  # remove duplicates, keep order
    print(f"\nUnique PMIDs: {len(unique_pmids)} (from {len(all_pmids)} total)")

    papers = fetch_abstracts(unique_pmids)
    print(f"Papers with title + abstract: {len(papers)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    download_papers()