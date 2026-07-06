"""
routes_chat.py
===============
WHY THIS FILE EXISTS:
HTTP boundary for Part 7 (chat pipeline): question -> embed -> Pinecone
search -> top-k chunks -> prompt -> Groq -> answer. Also persists chat
history so `chat_history.retrieved_chunk_ids` gives you a debuggable audit
trail of exactly which chunks grounded any given answer.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.models import ChatSession, ChatHistory
from app.db.session import get_db
from app.retrieval.llm_service import generate_answer
from app.retrieval.prompt_builder import build_prompt
from app.retrieval.retriever import retrieve_relevant_chunks

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None
    document_id: str | None = None  # optional: scope retrieval to one document


@router.post("/ask")
def ask_question(req: ChatRequest, db: Session = Depends(get_db)):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    session_id = req.session_id
    if not session_id:
        session = ChatSession()
        db.add(session)
        db.flush()
        session_id = str(session.session_id)

    matches = retrieve_relevant_chunks(req.question, document_id=req.document_id)
    prompt = build_prompt(req.question, matches)
    answer = generate_answer(prompt)

    chunk_ids = [m["chunk_id"] for m in matches]

    db.add(ChatHistory(session_id=session_id, role="user", content=req.question, retrieved_chunk_ids=chunk_ids))
    db.add(ChatHistory(session_id=session_id, role="assistant", content=answer, retrieved_chunk_ids=chunk_ids))
    db.commit()

    return {
        "session_id": session_id,
        "answer": answer,
        "sources": [{"chunk_id": m["chunk_id"], "page": m["metadata"].get("page"), "score": m["score"]} for m in matches],
    }
