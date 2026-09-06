# Gastric Cancer Case RAG — Local Chat App

A retrieval-augmented generation (RAG) chat app over gastric cancer case reports
(sourced from the MultiCaRe / PMC Open Access dataset). Hybrid retrieval
(BM25 + vector search, combined with Reciprocal Rank Fusion) followed by
cross-encoder re-ranking, then answer generation with source citations.
ChatGPT-style local web UI with session history — no login required.

## ⚠️ Before you start: you need your data files

This repo ships with code only, **not** the actual embeddings — those were
generated in your Kaggle notebook and are specific to your filtered dataset.

From your Kaggle notebook, make sure you've run:

```python
chunks_df.drop(columns=['embedding']).to_parquet('/kaggle/working/chunks_metadata.parquet', index=False)
np.save('/kaggle/working/chunk_embeddings.npy', np.vstack(chunks_df['embedding'].values))
```

Download both files from the Kaggle output panel and place them here:

```
data/chunks_metadata.parquet
data/chunk_embeddings.npy
```

The app will refuse to start with a clear error message if these are missing.

(The local Qdrant vector index is built automatically from these two files
the first time you run the app — you don't need to transfer the Qdrant
folder from Kaggle.)

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your OpenAI key:

```bash
cp .env.example .env
# then edit .env and paste your real key
```

Load the key before running (or use python-dotenv / your shell profile):

```bash
export OPENAI_API_KEY="sk-..."          # macOS/Linux
set OPENAI_API_KEY=sk-...               # Windows (cmd)
```

## Run

```bash
uvicorn api.main:app --reload --port 8000
```

Open **http://localhost:8000** — the FastAPI backend serves the frontend
directly, so there's no separate frontend server to run.

## Project structure

```
rag-stomach-cancer/
├── data/
│   ├── chunks_metadata.parquet   # <- you provide this (see above)
│   └── chunk_embeddings.npy      # <- you provide this (see above)
├── src/
│   └── rag_pipeline.py           # retrieval + reranking + generation logic
├── api/
│   ├── main.py                   # FastAPI app + routes
│   └── database.py               # SQLite chat history (sessions/messages)
├── frontend/
│   └── index.html                # ChatGPT-style chat UI (vanilla JS)
├── qdrant_data/                  # auto-created local vector index
├── chat_history.db               # auto-created SQLite DB
├── requirements.txt
├── .env.example
└── .gitignore
```

## How it works

1. **Retrieval**: query is run through both BM25 (keyword) and vector search
   (via a local Qdrant instance), then combined using Reciprocal Rank Fusion.
2. **Re-ranking**: the fused candidate set is re-scored with a cross-encoder
   (`bge-reranker-base`) to sharpen relevance beyond what embedding similarity
   alone captures.
3. **Generation**: the top re-ranked chunks are passed to an OpenAI model as
   context, with a system prompt that requires the model to cite `case_id`
   for every claim and to say so explicitly if the context doesn't contain
   an answer.
4. **Chat history**: sessions and messages are stored in a local SQLite file
   (`chat_history.db`), no login/auth — fine for a local solo demo.

## Known limitations (worth stating explicitly in a portfolio writeup)

- Retrieval is based only on the latest user message, not the full
  conversation — a good documented "future improvement."
- No user separation: all sessions are visible to whoever opens the app.
  Fine for local/solo use, not meant for multi-user deployment as-is.
- This is a research/demo project, not a clinical tool — do not use for
  actual medical decision-making.

## Suggested next steps

- Add streaming responses (token-by-token) for a more ChatGPT-like feel.
- Add an eval script (see your Kaggle notebook's `eval_set.json` +
  `evaluate_retrieval()`) as a `scripts/run_eval.py` for reproducible
  before/after retrieval comparisons.
- Containerize with Docker for easier setup/deployment.
