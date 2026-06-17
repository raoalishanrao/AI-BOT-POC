"""Persist counseling sessions and messages in Supabase."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import config
from src.chat.profile import StudentProfile
from src.ingest.supabase_client import get_supabase_client
from src.utils.logger import setup_logger

log = setup_logger("chat.session")

_memory_sessions: dict[str, dict[str, Any]] = {}


@dataclass
class SessionState:
    session_id: str
    user_id: str | None = None
    profile: StudentProfile = field(default_factory=StudentProfile)
    stage: str = "introduction"
    lead_status: str = "new"
    recommended_programs: list[str] = field(default_factory=list)
    messages: list[dict[str, str]] = field(default_factory=list)
    lead_score: int = 0
    contact_nudges: int = 0
    persisted: bool = False

    def to_row(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "profile": self.profile.to_dict(),
            "stage": self.stage,
            "lead_status": self.lead_status,
            "recommended_programs": self.recommended_programs,
        }


def _use_supabase() -> bool:
    return bool(config.SUPABASE_URL and (config.SUPABASE_SERVICE_KEY or config.SUPABASE_ANON_KEY))


def create_session(device_info: dict[str, Any] | None = None) -> SessionState:
    session_id = str(uuid.uuid4())
    state = SessionState(session_id=session_id)

    if _use_supabase():
        try:
            client = get_supabase_client()
            full_row = {
                "session_id": session_id,
                "device_info": device_info or {},
                "profile": {},
                "stage": "introduction",
                "lead_status": "new",
                "recommended_programs": [],
            }
            try:
                client.table("chat_sessions").insert(full_row).execute()
            except Exception:
                client.table("chat_sessions").insert(
                    {"session_id": session_id, "device_info": device_info or {}}
                ).execute()
            state.persisted = True
        except Exception as exc:
            log.warning("Supabase session create failed, using memory: %s", exc)

    _memory_sessions[session_id] = _state_to_memory(state)
    return state


def get_session(session_id: str) -> SessionState | None:
    if session_id in _memory_sessions:
        return _memory_to_state(_memory_sessions[session_id])

    if not _use_supabase():
        return None

    try:
        client = get_supabase_client()
        row = (
            client.table("chat_sessions")
            .select("*")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
        ).data
        if not row:
            return None

        messages = (
            client.table("chat_messages")
            .select("role, content")
            .eq("session_id", session_id)
            .order("created_at")
            .execute()
        ).data or []

        state = SessionState(
            session_id=session_id,
            user_id=row[0].get("user_id"),
            profile=StudentProfile.from_dict(row[0].get("profile")),
            stage=row[0].get("stage") or "introduction",
            lead_status=row[0].get("lead_status") or "new",
            recommended_programs=row[0].get("recommended_programs") or [],
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            persisted=True,
        )
        _memory_sessions[session_id] = _state_to_memory(state)
        return state
    except Exception as exc:
        log.warning("Supabase session load failed: %s", exc)
        return None


def save_session(state: SessionState) -> None:
    _memory_sessions[state.session_id] = _state_to_memory(state)
    if not _use_supabase() or not state.persisted:
        return

    try:
        client = get_supabase_client()
        payload = {
            "profile": state.profile.to_dict(),
            "stage": state.stage,
            "lead_status": state.lead_status,
            "recommended_programs": state.recommended_programs,
        }
        if state.user_id:
            payload["user_id"] = state.user_id
        try:
            client.table("chat_sessions").update(payload).eq("session_id", state.session_id).execute()
        except Exception:
            pass
    except Exception as exc:
        log.warning("Supabase session save failed: %s", exc)


def append_message(state: SessionState, role: str, content: str) -> None:
    state.messages.append({"role": role, "content": content})
    if len(state.messages) > config.CHAT_HISTORY_LIMIT * 2:
        state.messages = state.messages[-config.CHAT_HISTORY_LIMIT * 2 :]

    _memory_sessions[state.session_id] = _state_to_memory(state)

    if not _use_supabase() or not state.persisted:
        return

    try:
        client = get_supabase_client()
        client.table("chat_messages").insert(
            {"session_id": state.session_id, "role": role, "content": content}
        ).execute()
    except Exception as exc:
        log.warning("Supabase message save failed: %s", exc)


def save_lead(lead: dict[str, Any]) -> None:
    """Deprecated: use src.chat.users.upsert_counselor_lead instead."""
    if not _use_supabase():
        log.info("Lead captured (memory only): %s", lead)
        return

    try:
        client = get_supabase_client()
        session_id = lead.get("session_id")
        if session_id:
            existing = (
                client.table("counselor_leads")
                .select("id")
                .eq("session_id", session_id)
                .limit(1)
                .execute()
            ).data
            if existing:
                client.table("counselor_leads").update(lead).eq("session_id", session_id).execute()
                return
        client.table("counselor_leads").insert(lead).execute()
        log.info("Lead saved for session %s", lead.get("session_id"))
    except Exception as exc:
        log.warning("Supabase lead save failed: %s — lead=%s", exc, lead)


def _state_to_memory(state: SessionState) -> dict[str, Any]:
    return {
        "session_id": state.session_id,
        "user_id": state.user_id,
        "profile": state.profile.to_dict(),
        "stage": state.stage,
        "lead_status": state.lead_status,
        "recommended_programs": state.recommended_programs,
        "messages": list(state.messages),
        "lead_score": state.lead_score,
        "contact_nudges": state.contact_nudges,
        "persisted": state.persisted,
    }


def _memory_to_state(data: dict[str, Any]) -> SessionState:
    return SessionState(
        session_id=data["session_id"],
        user_id=data.get("user_id"),
        profile=StudentProfile.from_dict(data.get("profile")),
        stage=data.get("stage", "introduction"),
        lead_status=data.get("lead_status", "new"),
        recommended_programs=data.get("recommended_programs") or [],
        messages=list(data.get("messages") or []),
        lead_score=int(data.get("lead_score") or 0),
        contact_nudges=int(data.get("contact_nudges") or 0),
        persisted=bool(data.get("persisted")),
    )
