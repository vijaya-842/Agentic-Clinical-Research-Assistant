"""Compare Dense vs BM25 vs Hybrid (RRF) retrieval of trials - no LLM, so it is fast.

For every question with gold trial IDs (main + hard sets) we rank trials three ways
and report Recall@5, Recall@10, Precision@5, Hit@5 and MRR.

Run from the project root:
    python -m scripts.retrieval_experiment
"""
import json
from pathlib import Path

from src.evaluation.retrieval_metrics import average, hit_at_k, mrr, precision_at_k, recall_at_k
from src.reranking.cross_encoder import text_of
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import rrf_fuse, to_trial_ranking

QUESTION_FILES = [Path("eval/questions.json"), Path("eval/questions_hard.json")]
REPORT = Path("reports/retrieval_experiment.md")
RESULTS = Path("eval/retrieval_experiment.json")
CANDIDATES = 50  # chunks per method before turning them into a trial ranking


def load_questions():
    questions = []
    for path in QUESTION_FILES:
        if path.exists():
            data = json.load(open(path, encoding="utf-8"))
            questions += [q for q in (data["questions"] if isinstance(data, dict) else data) if q.get("gold_nct_ids")]
    return questions


def score(ranked, gold):
    return {
        "recall@5": recall_at_k(ranked, gold, 5),
        "recall@10": recall_at_k(ranked, gold, 10),
        "precision@5": precision_at_k(ranked, gold, 5),
        "hit@5": hit_at_k(ranked, gold, 5),
        "mrr": mrr(ranked, gold),
    }


def main():
    from src.retrieval.dense import DenseRetriever

    print("Loading dense index and building/loading BM25 (first time takes a few minutes)...")
    dense = DenseRetriever.from_disk()
    bm25 = BM25Index.load_or_build()

    rows, per_question = [], []
    for q in load_questions():
        dense_trials = to_trial_ranking(text_of(r) for r in dense.search(q["question"], top_k=CANDIDATES))
        bm25_trials = to_trial_ranking(chunk_id for chunk_id, _, _ in bm25.search(q["question"], top_k=CANDIDATES))
        hybrid_trials = rrf_fuse([dense_trials, bm25_trials])
        gold = q["gold_nct_ids"]
        scores = {name: score(ranked, gold) for name, ranked in
                  [("Dense", dense_trials), ("BM25", bm25_trials), ("Hybrid (RRF)", hybrid_trials)]}
        per_question.append({"id": q["id"], "gold": len(gold), "scores": scores})
        print(f"{q['id']:4} gold={len(gold):2}  " + "  ".join(
            f"{name}: R@10={s['recall@10']:.2f} MRR={s['mrr']:.2f}" for name, s in scores.items()))

    lines = ["# Retrieval experiment: Dense vs BM25 vs Hybrid", "",
             f"Questions with gold trials: {len(per_question)} (no metadata filters, no reranker)", "",
             "| Method | Recall@5 | Recall@10 | Precision@5 | Hit@5 | MRR |", "|---|---|---|---|---|---|"]
    for name in ["Dense", "BM25", "Hybrid (RRF)"]:
        m = {metric: average([p["scores"][name][metric] for p in per_question])
             for metric in ["recall@5", "recall@10", "precision@5", "hit@5", "mrr"]}
        lines.append(f"| {name} | {m['recall@5']} | {m['recall@10']} | {m['precision@5']} | {m['hit@5']} | {m['mrr']} |")
    report = "\n".join(lines)
    print("\n" + report)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report + "\n", encoding="utf-8")
    json.dump(per_question, open(RESULTS, "w", encoding="utf-8"), indent=2)
    print(f"\nSaved {REPORT} and {RESULTS}")


if __name__ == "__main__":
    main()