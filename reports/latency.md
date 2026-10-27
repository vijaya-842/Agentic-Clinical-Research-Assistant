# Latency report

Setup: keyword router, dense search + MedCPT reranker, qwen2.5:3b via Ollama, CPU.
The first AI question also includes loading the LLM into memory (warm-up).

- Model + data loading (once at start): **15.5 s**
- Total for 12 questions: **786.8 s**

## By tool

| Tool | Questions | Average (s) | Slowest (s) |
|---|---|---|---|
| small_talk | 1 | 0.00 | 0.00 |
| safety_refusal | 1 | 0.00 | 0.00 |
| trial_lookup | 1 | 0.00 | 0.00 |
| count_trials | 1 | 0.09 | 0.09 |
| advanced_rag | 3 | 62.76 | 81.64 |
| literature_rag | 1 | 97.77 | 97.77 |
| list_rag | 4 | 125.17 | 163.08 |

## Per question

| # | Tool | Seconds | Question |
|---|---|---|---|
| 1 | advanced_rag | 81.64 | Which clinical trial tests thymosin alpha 1 after surgery for high-risk stage II and III colorectal cancer? Give the NCT ID, the sponsor and the primary outcome. |
| 2 | advanced_rag | 40.17 | Which trial combines capecitabine, tucatinib and intrathecal trastuzumab for breast cancer patients with leptomeningeal disease? Give the NCT ID, the sponsor and the primary outcome. |
| 3 | advanced_rag | 66.46 | Is there a trial testing dapagliflozin to prevent heart damage from anthracycline chemotherapy in breast cancer patients? Give the NCT ID, the sponsor and the primary outcome. |
| 4 | list_rag | 105.99 | Find recruiting Phase 3 clinical trials in lung cancer. |
| 5 | list_rag | 68.71 | Find recruiting clinical trials testing pembrolizumab in melanoma. |
| 6 | list_rag | 163.08 | List clinical trials that include osimertinib as an intervention. |
| 7 | list_rag | 162.92 | Which non-small cell lung cancer trials exclude patients with brain metastases? |
| 8 | count_trials | 0.09 | How many recruiting pancreatic cancer trials are in the downloaded ClinicalTrials.gov dataset? |
| 9 | literature_rag | 97.77 | What is the mechanism of action of osimertinib? |
| 10 | safety_refusal | 0.00 | My father has stage IV lung cancer. Which treatment should he take? |
| 11 | small_talk | 0.00 | hi |
| 12 | trial_lookup | 0.00 | Tell me about NCT05086614 |
