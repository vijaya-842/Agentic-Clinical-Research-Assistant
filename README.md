# Agentic Clinical Research Assistant

A biomedical research assistant built using Retrieval-Augmented Generation (RAG), structured clinical trial data, and agentic AI workflows.

> **Research use only. Not medical advice.**

## Project Goal

The system answers biomedical and oncology clinical trial questions using retrieved evidence from PubMed literature and ClinicalTrials.gov data, with citations to PMIDs and NCT IDs.

The project implements and compares:

1. LLM-only baseline
2. Naive RAG
3. Advanced RAG with hybrid retrieval and reranking
4. Agentic RAG with dynamic tool selection and verification

## Core Technologies

- Python 3.11
- Open-source local LLMs via Ollama (no proprietary hosted LLMs)
- LangGraph
- Hugging Face / Sentence Transformers
- FAISS / Qdrant
- PostgreSQL
- ClinicalTrials.gov API v2 and PubMed E-utilities
- FastAPI
- Streamlit
- Docker

## Setup

```powershell
git clone https://github.com/vijaya-842/Agentic-Clinical-Research-Assistant.git
cd Agentic-Clinical-Research-Assistant
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen2.5:3b
```

## Hardware Used

- Intel i5-1235U, 16 GB RAM, Intel Iris Xe (CPU only)
- Generator model: `qwen2.5:3b` via Ollama

## Project Status

- [x] Project structure and configuration
- [x] Local LLM client (Ollama)
- [x] LLM-only baseline
- [x] Document ingestion and chunking pipeline
- [x] PubMed fetcher
- [ ] ClinicalTrials.gov fetcher
- [ ] Embeddings and vector index
- [ ] Naive RAG
- [ ] Advanced RAG
- [ ] Agentic RAG
- [ ] Evaluation
- [ ] UI and Docker