"""FastAPI server for the embeddable counseling chat widget."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from src.chat.counselor import chat, start_session
from src.chat.llm import active_chat_model

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Iqra University AI Counselor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class StartSessionRequest(BaseModel):
    device_info: dict | None = None


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1)


@app.get("/")
def root():
    widget = STATIC_DIR / "widget.html"
    if widget.exists():
        return FileResponse(widget)
    return {"status": "ok", "message": "POST /session/start then POST /chat"}


@app.post("/session/start")
def session_start(body: StartSessionRequest | None = None):
    device_info = body.device_info if body else None
    response = start_session(device_info=device_info)
    return {
        "session_id": response.session_id,
        "reply": response.reply,
        "stage": response.stage,
        "lead_status": response.lead_status,
        "ctas": response.ctas,
    }


@app.post("/chat")
def chat_message(body: ChatRequest):
    try:
        response = chat(body.session_id, body.message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "session_id": response.session_id,
        "reply": response.reply,
        "stage": response.stage,
        "lead_status": response.lead_status,
        "profile": response.profile,
        "recommended_programs": response.recommended_programs,
        "ctas": response.ctas,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "provider": config.CHAT_PROVIDER,
        "model": active_chat_model(),
        "embeddings": config.EMBEDDING_MODEL,
    }
