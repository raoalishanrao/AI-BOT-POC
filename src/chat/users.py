"""Create and link chat_users from counseling profiles."""

from __future__ import annotations

import hashlib
from typing import Any

import config
from src.chat.leads import build_lead_record
from src.chat.profile import StudentProfile
from src.ingest.supabase_client import get_supabase_client
from src.utils.logger import setup_logger

log = setup_logger("chat.users")


def _synthetic_phone_for_email(email: str) -> str:
    digest = hashlib.md5(email.lower().encode()).hexdigest()[:10]
    return f"E{digest}"


def _contact_phone(profile: StudentProfile) -> str | None:
    if profile.phone:
        return profile.phone
    if profile.email:
        return _synthetic_phone_for_email(profile.email)
    return None


def _use_supabase() -> bool:
    return bool(config.SUPABASE_URL and (config.SUPABASE_SERVICE_KEY or config.SUPABASE_ANON_KEY))


def _find_user_id(client, profile: StudentProfile) -> str | None:
    if profile.phone:
        rows = (
            client.table("chat_users")
            .select("id")
            .eq("phone_number", profile.phone)
            .limit(1)
            .execute()
        ).data
        if rows:
            return rows[0]["id"]

    if profile.email:
        rows = (
            client.table("chat_users")
            .select("id")
            .eq("email", profile.email)
            .limit(1)
            .execute()
        ).data
        if rows:
            return rows[0]["id"]

    return None


def upsert_chat_user(profile: StudentProfile) -> str | None:
    """Create or update chat_users when we have contact info."""
    if not _use_supabase():
        return None
    if not profile.name and not profile.email and not profile.phone:
        return None

    client = get_supabase_client()
    name = profile.name or "Prospective Student"
    phone = _contact_phone(profile)
    payload: dict[str, Any] = {
        "name": name,
        "email": profile.email,
        "phone_number": phone,
    }

    try:
        existing_id = _find_user_id(client, profile)
        if existing_id:
            client.table("chat_users").update(
                {k: v for k, v in payload.items() if v is not None}
            ).eq("id", existing_id).execute()
            log.info("Updated chat_user %s", existing_id)
            return existing_id

        if not phone and not profile.email:
            return None

        row = {k: v for k, v in payload.items() if v is not None}
        row.setdefault("name", name)
        inserted = client.table("chat_users").insert(row).execute()
        user_id = inserted.data[0]["id"]
        log.info("Created chat_user %s", user_id)
        return user_id
    except Exception as exc:
        log.warning("chat_users upsert failed: %s", exc)
        return None


def link_session_user(session_id: str, user_id: str) -> None:
    if not _use_supabase():
        return
    try:
        client = get_supabase_client()
        client.table("chat_sessions").update({"user_id": user_id}).eq(
            "session_id", session_id
        ).execute()
    except Exception as exc:
        log.warning("Failed to link session %s to user %s: %s", session_id, user_id, exc)


def upsert_counselor_lead(
    session_id: str,
    profile: StudentProfile,
    *,
    lead_score: int,
    lead_status: str,
    recommended_programs: list[str],
) -> None:
    """Save or update a row in counselor_leads for admissions follow-up."""
    if not _use_supabase():
        return
    if not (profile.name or profile.email or profile.phone):
        return

    programs = recommended_programs or profile.interested_programs
    notes = f"Status: {lead_status}"
    if programs:
        notes += f" | Programs: {', '.join(programs[:5])}"

    record = build_lead_record(session_id, profile, lead_score, notes=notes)

    try:
        client = get_supabase_client()
        existing = (
            client.table("counselor_leads")
            .select("id")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
        ).data

        if existing:
            client.table("counselor_leads").update(record).eq("session_id", session_id).execute()
            log.info("Updated counselor_lead for session %s", session_id)
        else:
            client.table("counselor_leads").insert(record).execute()
            log.info("Created counselor_lead for session %s", session_id)
    except Exception as exc:
        log.warning("counselor_leads upsert failed: %s — %s", exc, record)


def sync_session_user_and_lead(state) -> None:
    """Link session to chat_users and upsert counselor_leads when profile has contact info."""
    user_id = upsert_chat_user(state.profile)
    if user_id:
        state.user_id = user_id
        link_session_user(state.session_id, user_id)

    if state.lead_status in {"warm", "interested", "captured"} or state.profile.lead_contact_ready():
        upsert_counselor_lead(
            state.session_id,
            state.profile,
            lead_score=state.lead_score,
            lead_status=state.lead_status,
            recommended_programs=state.recommended_programs,
        )
