# Retrieval experiment: Dense vs BM25 vs Hybrid

Questions with gold trials: 12 (no metadata filters, no reranker)

| Method | Recall@5 | Recall@10 | Precision@5 | Hit@5 | MRR |
|---|---|---|---|---|---|
| Dense | 0.449 | 0.474 | 0.267 | 0.75 | 0.575 |
| BM25 | 0.428 | 0.58 | 0.217 | 0.583 | 0.497 |
| Hybrid (RRF) | 0.387 | 0.572 | 0.233 | 0.667 | 0.598 |
