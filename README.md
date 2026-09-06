# Gastric Cancer Case RAG — Retrieval-Augmented Chat over Clinical Case Reports

A retrieval-augmented generation (RAG) system built over real, de-identified gastric cancer case reports, with a ChatGPT-style local chat interface. Built to go beyond tutorial-level RAG — this project implements hybrid retrieval, cross-encoder re-ranking, and a measurable eval pipeline to demonstrate *why* naive RAG fails and how each component improves retrieval quality.

## Why this project

Most RAG tutorials stop at "chunk text → embed → vector search → done." That pipeline breaks down on real-world data: keyword-heavy queries get missed by pure vector search, naive fixed-size chunking splits sentences and tables mid-thought, and vector similarity alone often surfaces plausible-sounding but wrong results. This project was built specifically to confront those failure modes and fix them, with before/after evidence.

## Features

- **Real, messy source data**: gastric/stomach cancer case reports sourced from the [MultiCaRe dataset](https://www.kaggle.com/datasets/mauronievasoffidani/multicare) (built from PMC Open Access case reports) — genuine clinical narrative text, not a clean toy dataset.
- **Adaptive chunking**: short cases are kept whole; only long cases are split, using token-aware sentence-boundary chunking with overlap — avoiding the common mistake of fragmenting short documents unnecessarily.
- **Hybrid retrieval**: combines BM25 (keyword/sparse) and dense vector search (via Qdrant), fused with Reciprocal Rank Fusion (RRF), so retrieval doesn't fail on queries that are keyword-heavy or vocabulary-mismatched.
- **Cross-encoder re-ranking**: a second-pass re-ranker (`bge-reranker-base`) re-scores retrieved candidates directly against the query, catching relevance nuances that embedding similarity alone misses.
- **Grounded generation with citations**: every answer is generated with explicit `case_id` citations back to source documents, and the model is instructed to say "not enough information" rather than guess.
- **Quantitative evaluation**: a hand-built eval set with known-correct source cases, scored for hit rate, precision, recall, and MRR — used to compare naive vector-only retrieval against the full hybrid + re-ranked pipeline.
- **Local chat UI with history**: a ChatGPT-style interface (session sidebar, persistent history, no login) built with FastAPI + vanilla JS, backed by local SQLite.

## Architecture

```
Case reports (MultiCaRe/PMC) → Adaptive chunking → Embeddings (bge-base-en-v1.5)
                                                          ↓
                                                   Qdrant vector store
                                                          ↓
Query → Hybrid Retrieval (BM25 + vector, RRF fusion) → Cross-encoder re-rank
                                                          ↓
                                          LLM generation with citations (OpenAI)
                                                          ↓
                                          FastAPI backend + local chat UI (SQLite history)
```

## Tech stack

| Layer | Tool |
|---|---|
| Embeddings | `sentence-transformers` (`BAAI/bge-base-en-v1.5`) |
| Sparse retrieval | `rank_bm25` |
| Vector store | Qdrant (local/embedded mode) |
| Re-ranking | `bge-reranker-base` (cross-encoder) |
| Generation | OpenAI API (`gpt-4o`) |
| Backend | FastAPI |
| Chat history | SQLite |
| Frontend | Vanilla JS/HTML/CSS |
| Prototyping | Kaggle Notebooks |

## Results

> Fill this in with your actual numbers from the eval harness — this is the section reviewers/interviewers will look at first.

| Retrieval strategy | Hit Rate | Precision | Recall | MRR |
|---|---|---|---|---|
| Vector-only (naive baseline) | – | – | – | – |
| Hybrid (BM25 + vector, RRF) | – | – | – | – |
| Hybrid + re-ranked (full pipeline) | – | – | – | – |

*(Run `evaluate_retrieval()` against `eval_set.json` for each strategy to populate this table — see the eval section in the project notebook.)*

## Setup

See [`SETUP.md`](./SETUP.md) for full local installation and run instructions.

Quick start:

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell; use `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
cp .env.example .env             # then add your OPENAI_API_KEY
uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000`.

## Known limitations

- Retrieval is based only on the latest user message, not the full conversation thread.
- No user authentication/separation — intended for local, single-user demo use.
- This is a research/portfolio demo, **not** a clinical decision-support tool.

## Future improvements

- [ ] Streaming token-by-token responses
- [ ] Conversation-aware query reformulation for multi-turn retrieval
- [ ] Docker containerization for one-command setup
- [ ] Expand corpus beyond gastric cancer to broader oncology case reports

## Acknowledgments

Built on the [MultiCaRe Dataset](https://zenodo.org/records/13936721) (Nievas Offidani, M. et al.), derived from PubMed Central Open Access case reports.
