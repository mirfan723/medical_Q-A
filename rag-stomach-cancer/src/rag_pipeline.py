import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI

# ---- Load data ----
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks_metadata.parquet")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "chunk_embeddings.npy")

if not os.path.exists(CHUNKS_PATH) or not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError(
        "Missing data files. Copy 'chunks_metadata.parquet' and 'chunk_embeddings.npy' "
        "(generated in your Kaggle notebook) into the 'data/' folder before running the app. "
        "See README.md for details."
    )

chunks_df = pd.read_parquet(CHUNKS_PATH)
embeddings_matrix = np.load(EMBEDDINGS_PATH)
chunks_df["embedding"] = list(embeddings_matrix)

# ---- Load models (once, at startup) ----
embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
reranker = CrossEncoder("BAAI/bge-reranker-base")
llm_client = OpenAI()  # picks up OPENAI_API_KEY from environment

# ---- Vector store ----
QDRANT_PATH = os.path.join(os.path.dirname(__file__), "..", "qdrant_data")
qdrant = QdrantClient(path=QDRANT_PATH)

COLLECTION = "gastric_cancer_cases"
if not qdrant.collection_exists(COLLECTION):
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=embeddings_matrix.shape[1], distance=Distance.COSINE),
    )
    points = [
        PointStruct(
            id=i,
            vector=row["embedding"].tolist(),
            payload={
                "case_id": row["case_id"],
                "chunk_index": int(row["chunk_index"]),
                "chunk_text": row["chunk_text"],
            },
        )
        for i, row in chunks_df.reset_index(drop=True).iterrows()
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)


# ---- BM25 (keyword / sparse retrieval) ----
def simple_tokenize(text):
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return text.split()


tokenized_corpus = [simple_tokenize(t) for t in chunks_df["chunk_text"].tolist()]
bm25 = BM25Okapi(tokenized_corpus)


def bm25_search(query, top_k=10):
    scores = bm25.get_scores(simple_tokenize(query))
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


def vector_search(query, top_k=10):
    query_vec = embed_model.encode(f"query: {query}", normalize_embeddings=True)
    results = qdrant.query_points(collection_name=COLLECTION, query=query_vec.tolist(), limit=top_k)
    return [(r.id, r.score) for r in results.points]


def reciprocal_rank_fusion(result_lists, k=60, top_k=10):
    """Combine ranked result lists using Reciprocal Rank Fusion (RRF)."""
    fused_scores = {}
    for result_list in result_lists:
        for rank, (doc_id, _) in enumerate(result_list):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


def hybrid_search(query, top_k=10, candidate_k=25):
    """BM25 + vector search fused with RRF."""
    bm25_results = bm25_search(query, top_k=candidate_k)
    vector_results = vector_search(query, top_k=candidate_k)
    return reciprocal_rank_fusion([bm25_results, vector_results], top_k=top_k)


def rerank(query, candidate_ids, top_k=5):
    """Cross-encoder re-ranking over the fused candidate set."""
    candidate_texts = [chunks_df.iloc[cid]["chunk_text"] for cid, _ in candidate_ids]
    pairs = [[query, text] for text in candidate_texts]
    scores = reranker.predict(pairs)
    reranked = sorted(
        zip([cid for cid, _ in candidate_ids], candidate_texts, scores),
        key=lambda x: x[2],
        reverse=True,
    )
    return reranked[:top_k]


def build_context_block(reranked_chunks):
    blocks = []
    for i, (idx, text, score) in enumerate(reranked_chunks, start=1):
        case_id = chunks_df.iloc[idx]["case_id"]
        blocks.append(f"[Source {i} | case_id={case_id}]\n{text}")
    return "\n\n".join(blocks)


SYSTEM_PROMPT = (
    "You are a clinical literature assistant. Answer the user's question "
    "using ONLY the provided case excerpts. For every claim, cite the source "
    "using its case_id in brackets, e.g. [PMC1234567]. If the excerpts don't "
    "contain enough information to answer, say so explicitly rather than guessing. "
    "This is a research demo, not a source of medical advice."
)


def generate_answer(query, conversation_history=None, top_k=5, model="gpt-4o"):
    """
    Full RAG pipeline: hybrid retrieval -> rerank -> LLM generation with citations.

    conversation_history: optional list of {"role": "user"/"assistant", "content": str}
    dicts from earlier turns in the same session, used to give the LLM conversational
    context. Note: retrieval itself is still based only on the latest query text.
    """
    fused = hybrid_search(query, top_k=15, candidate_k=25)
    reranked = rerank(query, fused, top_k=top_k)
    context = build_context_block(reranked)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})

    response = llm_client.chat.completions.create(model=model, max_tokens=600, messages=messages)
    answer = response.choices[0].message.content
    sources = list(dict.fromkeys(chunks_df.iloc[idx]["case_id"] for idx, _, _ in reranked))

    return {"answer": answer, "sources": sources}
