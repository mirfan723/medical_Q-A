from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import database as db
from src.rag_pipeline import generate_answer

app = FastAPI(title="Gastric Cancer RAG Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev; tighten if you ever deploy
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init_db()


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/api/sessions")
def new_session():
    session_id = db.create_session()
    return {"session_id": session_id}


@app.get("/api/sessions")
def get_sessions():
    return db.list_sessions()


@app.get("/api/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    return db.get_messages(session_id)


@app.delete("/api/sessions/{session_id}")
def remove_session(session_id: str):
    db.delete_session(session_id)
    return {"ok": True}


@app.post("/api/chat")
def chat(req: ChatRequest):
    history = db.get_messages(req.session_id)
    conversation_history = [{"role": m["role"], "content": m["content"]} for m in history]

    db.add_message(req.session_id, "user", req.message)

    result = generate_answer(req.message, conversation_history=conversation_history)

    db.add_message(req.session_id, "assistant", result["answer"], result["sources"])

    # auto-title the session from the first message
    if len(history) == 0:
        title = req.message[:50] + ("..." if len(req.message) > 50 else "")
        db.rename_session(req.session_id, title)

    return {"answer": result["answer"], "sources": result["sources"]}


# serve the frontend (must be mounted last so /api routes above take priority)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
