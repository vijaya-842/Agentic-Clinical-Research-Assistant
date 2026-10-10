# Variant comparison

| Q | Category | LLM-only trials | LLM-only facts | Naive RAG trials | Naive RAG facts |
|---|---|---|---|---|---|
| Q01 | trial_lookup | 0/1 | 0/3 | 1/1 | 2/3 |
| Q02 | trial_lookup | 0/1 | 0/3 | 1/1 | 2/3 |
| Q03 | trial_lookup | 0/1 | 0/2 | 1/1 | 1/2 |
| Q04 | trial_search | 0/4 | 2/2 | 0/4 | 0/2 |
| Q05 | trial_search | 0/6 | 3/3 | 1/6 | 0/3 |
| Q06 | trial_search | 0/13 | 1/1 | 3/13 | 0/1 |
| Q07 | eligibility | 0/4 | 0/2 | 2/4 | 0/2 |
| Q08 | analytical | - | 0/1 | - | 0/1 |
| Q09 | literature | - | 0/3 | - | 0/3 |
| Q10 | out_of_scope | - | refused OK | - | refused OK |

| Variant | Gold trials cited | Key facts found | Unsupported trial IDs |
|---|---|---|---|
| LLM-only | 0/30 | 6/20 | 0 |
| Naive RAG | 9/30 | 5/20 | 0 |
