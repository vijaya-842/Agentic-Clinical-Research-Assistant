import requests
import xml.etree.ElementTree as ET

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def search_pubmed(
    query: str,
    max_results: int = 20,
) -> list[str]:
    """
    Search PubMed and return a list of PubMed IDs (PMIDs).
    """

    url = f"{BASE_URL}/esearch.fcgi"

    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }

    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    pmids = data["esearchresult"]["idlist"]

    return pmids

def fetch_pubmed_articles(
    pmids: list[str],
) -> str:
    """
    Fetch PubMed article records for a list of PMIDs.
    """

    url = f"{BASE_URL}/efetch.fcgi"

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }

    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    return response.text

def parse_pubmed_articles(
    articles_xml: str,
) -> list[dict]:
    """
    Parse PubMed XML and extract article information.
    """

    root = ET.fromstring(articles_xml)

    articles = []

    for article in root.findall("PubmedArticle"):

        pmid_element = article.find(".//PMID")
        title_element = article.find(".//ArticleTitle")

        abstract_elements = article.findall(
            ".//AbstractText"
        )

        pmid = (
            pmid_element.text
            if pmid_element is not None
            else None
        )

        title = (
            "".join(title_element.itertext()).strip()
            if title_element is not None
            else ""
        )

        abstract_parts = []

        for abstract_element in abstract_elements:

            abstract_text = "".join(
                abstract_element.itertext()
            ).strip()

            if abstract_text:
                abstract_parts.append(
                    abstract_text
                )

        abstract = "\n".join(abstract_parts)

        articles.append(
            {
                "id": pmid,
                "title": title,
                "abstract": abstract,
            }
        )

    return articles

if __name__ == "__main__":

    query = "Type 2 diabetes"

    print(f"\nSearching PubMed for: {query}")

    pmids = search_pubmed(
        query=query,
        max_results=5,
    )

    print(f"\nFound {len(pmids)} PubMed IDs:")

    for pmid in pmids:
        print(pmid)

    print("\nFetching article records...")

    articles_xml = fetch_pubmed_articles(pmids)

    print("\nParsing article records...")

    articles = parse_pubmed_articles(articles_xml)

    print(f"\nParsed {len(articles)} articles:\n")

    for article in articles:

        print("=" * 60)
        print(f"PMID: {article['id']}")
        print(f"Title: {article['title']}")

        print("\nAbstract:")

        if article["abstract"]:
            print(article["abstract"][:500])
        else:
            print("No abstract available.")