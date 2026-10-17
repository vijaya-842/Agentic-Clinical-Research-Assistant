from src.ingestion.pubmed_fetcher import build_search_terms, parse_pubmed_xml

SAMPLE_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <Journal>
          <JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue>
          <Title>Journal of Cancer Research</Title>
        </Journal>
        <ArticleTitle>Osimertinib: a <i>third-generation</i> EGFR inhibitor</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">EGFR mutations drive NSCLC.</AbstractText>
          <AbstractText Label="RESULTS">Osimertinib irreversibly binds mutant EGFR.</AbstractText>
        </Abstract>
        <PublicationTypeList><PublicationType>Review</PublicationType></PublicationTypeList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>99999999</PMID>
      <Article>
        <Journal>
          <JournalIssue><PubDate><MedlineDate>2019 Jan-Feb</MedlineDate></PubDate></JournalIssue>
          <Title>Some Journal</Title>
        </Journal>
        <ArticleTitle>A paper without an abstract</ArticleTitle>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_parses_title_with_italic_text():
    papers = parse_pubmed_xml(SAMPLE_XML)
    assert papers[0]["title"] == "Osimertinib: a third-generation EGFR inhibitor"


def test_joins_abstract_parts_with_labels():
    abstract = parse_pubmed_xml(SAMPLE_XML)[0]["abstract"]
    assert abstract == "BACKGROUND: EGFR mutations drive NSCLC. RESULTS: Osimertinib irreversibly binds mutant EGFR."


def test_reads_metadata():
    paper = parse_pubmed_xml(SAMPLE_XML)[0]
    assert paper["pmid"] == "12345678"
    assert paper["year"] == "2020"
    assert paper["journal"] == "Journal of Cancer Research"
    assert paper["publication_types"] == ["Review"]
    assert paper["url"] == "https://pubmed.ncbi.nlm.nih.gov/12345678/"


def test_skips_papers_without_abstract():
    papers = parse_pubmed_xml(SAMPLE_XML)
    assert len(papers) == 1


def test_search_terms_cover_cancers_and_drugs():
    terms = [t for t, _ in build_search_terms()]
    assert len(terms) == 35
    assert any("osimertinib" in t for t in terms)
    assert any("pancreatic cancer" in t for t in terms)