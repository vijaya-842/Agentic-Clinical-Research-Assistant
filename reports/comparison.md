# Variant comparison

| Q | Category | LLM-only trials | LLM-only facts | Naive RAG trials | Naive RAG facts | Adv. RAG (filters) trials | Adv. RAG (filters) facts | Adv. RAG (+ fact sheets) trials | Adv. RAG (+ fact sheets) facts | Agentic (+ count tool) trials | Agentic (+ count tool) facts | Agentic (+ list top-10) trials | Agentic (+ list top-10) facts | Agentic (+ code details) trials | Agentic (+ code details) facts | Agentic (+ PubMed) trials | Agentic (+ PubMed) facts | LangGraph agent (LLM picks tool) trials | LangGraph agent (LLM picks tool) facts | Agentic (+ reranker MiniLM) trials | Agentic (+ reranker MiniLM) facts | Agentic (+ reranker MedCPT) trials | Agentic (+ reranker MedCPT) facts | Agentic (+ MedCPT + guardrail) trials | Agentic (+ MedCPT + guardrail) facts | Agentic (+ hybrid BM25) trials | Agentic (+ hybrid BM25) facts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q01 | trial_lookup | 0/1 | 0/3 | 1/1 | 2/3 | 1/1 | 1/3 | 1/1 | 2/3 | 1/1 | 2/3 | 1/1 | 2/3 | 1/1 | 2/3 | 1/1 | 2/3 | 1/1 | 2/3 | 1/1 | 2/3 | 1/1 | 2/3 | 1/1 | 2/3 | 1/1 | 2/3 |
| Q02 | trial_lookup | 0/1 | 0/3 | 1/1 | 2/3 | 1/1 | 1/3 | 1/1 | 2/3 | 1/1 | 2/3 | 1/1 | 2/3 | 1/1 | 2/3 | 1/1 | 2/3 | 1/1 | 2/3 | 1/1 | 2/3 | 1/1 | 2/3 | 1/1 | 2/3 | 1/1 | 2/3 |
| Q03 | trial_lookup | 0/1 | 0/2 | 1/1 | 1/2 | 1/1 | 0/2 | 1/1 | 2/2 | 1/1 | 2/2 | 1/1 | 2/2 | 1/1 | 2/2 | 1/1 | 2/2 | 1/1 | 2/2 | 1/1 | 2/2 | 1/1 | 2/2 | 1/1 | 2/2 | 1/1 | 2/2 |
| Q04 | trial_search | 0/4 | 2/2 | 0/4 | 0/2 | 4/4 | 0/2 | 4/4 | 0/2 | 4/4 | 0/2 | 4/4 | 0/2 | 4/4 | 2/2 | 4/4 | 2/2 | 4/4 | 2/2 | 4/4 | 2/2 | 4/4 | 2/2 | 4/4 | 2/2 | 4/4 | 2/2 |
| Q05 | trial_search | 0/6 | 3/3 | 1/6 | 0/3 | 4/6 | 0/3 | 5/6 | 3/3 | 5/6 | 3/3 | 6/6 | 0/3 | 6/6 | 3/3 | 6/6 | 3/3 | 6/6 | 3/3 | 4/6 | 3/3 | 5/6 | 3/3 | 5/6 | 3/3 | 5/6 | 3/3 |
| Q06 | trial_search | 0/13 | 1/1 | 3/13 | 0/1 | 5/13 | 0/1 | 5/13 | 0/1 | 5/13 | 0/1 | 9/13 | 0/1 | 9/13 | 1/1 | 9/13 | 1/1 | 9/13 | 1/1 | 10/13 | 1/1 | 10/13 | 1/1 | 10/13 | 1/1 | 10/13 | 1/1 |
| Q07 | eligibility | 0/4 | 0/2 | 2/4 | 0/2 | 3/4 | 1/2 | 2/4 | 1/2 | 2/4 | 1/2 | 1/4 | 2/2 | 1/4 | 2/2 | 1/4 | 2/2 | 1/4 | 2/2 | 2/4 | 1/2 | 0/4 | 2/2 | 0/4 | 2/2 | 0/4 | 1/2 |
| Q08 | analytical | - | 0/1 | - | 0/1 | - | 0/1 | - | 0/1 | - | 1/1 | - | 1/1 | - | 1/1 | - | 1/1 | - | 1/1 | - | 1/1 | - | 1/1 | - | 1/1 | - | 1/1 |
| Q09 | literature | - | 2/3 | - | 0/3 | - | 1/3 | - | 0/3 | - | 0/3 | - | 0/3 | - | 0/3 | - | 3/3 | - | 3/3 | - | 2/3 | - | 3/3 | - | 3/3 | - | 3/3 |
| Q10 | out_of_scope | - | refused OK | - | refused OK | - | refused OK | - | refused OK | - | refused OK | - | refused OK | - | refused OK | - | refused OK | - | refused OK | - | no refusal | - | refused OK | - | refused OK | - | refused OK |

| Variant | Gold trials cited | Key facts found | Unsupported trial IDs |
|---|---|---|---|
| LLM-only | 0/30 | 8/20 | 0 |
| Naive RAG | 9/30 | 5/20 | 0 |
| Adv. RAG (filters) | 19/30 | 4/20 | 0 |
| Adv. RAG (+ fact sheets) | 19/30 | 10/20 | 0 |
| Agentic (+ count tool) | 19/30 | 11/20 | 0 |
| Agentic (+ list top-10) | 23/30 | 9/20 | 0 |
| Agentic (+ code details) | 23/30 | 15/20 | 0 |
| Agentic (+ PubMed) | 23/30 | 18/20 | 0 |
| LangGraph agent (LLM picks tool) | 23/30 | 18/20 | 0 |
| Agentic (+ reranker MiniLM) | 23/30 | 16/20 | 0 |
| Agentic (+ reranker MedCPT) | 22/30 | 18/20 | 0 |
| Agentic (+ MedCPT + guardrail) | 22/30 | 18/20 | 0 |
| Agentic (+ hybrid BM25) | 22/30 | 17/20 | 0 |
