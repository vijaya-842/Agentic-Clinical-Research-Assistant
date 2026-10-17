"""Search the PubMed paper index: question in, most similar papers out."""
import json

import faiss
import numpy as np

from src.embeddings.build_pubmed_index import INDEX_FILE, MODEL_NAME, PAPERS_OUT

# bge models work best when the QUESTION gets this prefix (papers do not)
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class PubMedRetriever:
    def __init__(self, model, index, papers):
        self.model = model
        self.index = index
        self.papers = papers

    @classmethod
    def from_disk(cls):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME)
        index = faiss.read_index(str(INDEX_FILE))
        with open(PAPERS_OUT, encoding="utf-8") as f:
            papers = json.load(f)
        return cls(model, index, papers)

    def search(self, question, top_k=5):
        """Return the top_k papers, best first, each with a similarity score."""
        vector = self.model.encode([QUERY_INSTRUCTION + question], normalize_embeddings=True)
        scores, positions = self.index.search(np.asarray(vector, dtype="float32"), top_k)
        results = []
        for score, pos in zip(scores[0], positions[0]):
            if pos == -1:  # FAISS returns -1 when there are fewer papers than top_k
                continue
            paper = dict(self.papers[pos])
            paper["score"] = round(float(score), 3)
            results.append(paper)
        return results


if __name__ == "__main__":
    retriever = PubMedRetriever.from_disk()
    question = "What is the mechanism of action of osimertinib?"
    print("Question:", question, "\n")
    for p in retriever.search(question, top_k=5):
        print(f"{p['score']}  PMID {p['pmid']} ({p['year']})  {p['title'][:100]}")