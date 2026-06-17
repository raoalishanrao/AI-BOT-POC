"""Lead qualification and capture."""

from __future__ import annotations

import re
from typing import Any

from src.chat.profile import StudentProfile

LEAD_SIGNALS: tuple[tuple[str, int], ...] = (
    (r"\bapply\b|\bapplication\b|\bhow (?:do|can) i apply\b", 3),
    (r"\badmission\b|\benroll\b|\bregister\b", 2),
    (r"\bfee\b|\bfees\b|\btuition\b|\bcost\b|\bafford\b", 2),
    (r"\bscholarship\b", 2),
    (r"\bdeadline\b|\bintake\b|\bwhen (?:does|can)\b", 2),
    (r"\bcall(?:back)?\b|\bcontact\b|\bspeak (?:to|with)\b|\badvisor\b|\bcounselor\b", 3),
    (r"\binquiry\b|\binquire\b|\binterested\b|\bsign me up\b", 3),
    (r"\bvisit\b|\bcampus tour\b", 1),
)

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(?:\+92|0)?3\d{2}[-\s]?\d{7}|\b0\d{2}[-\s]?\d{7,8}\b")


def detect_lead_score(message: str) -> int:
    score = 0
    for pattern, weight in LEAD_SIGNALS:
        if re.search(pattern, message, re.I):
            score += weight
    return score


def extract_contact_hints(message: str) -> dict[str, str]:
    hints: dict[str, str] = {}
    email = EMAIL_RE.search(message)
    if email:
        hints["email"] = email.group(0)
    phone = PHONE_RE.search(message)
    if phone:
        hints["phone"] = phone.group(0)
    return hints


def qualify_lead_status(profile: StudentProfile, message: str, current_status: str) -> str:
    score = detect_lead_score(message)
    if profile.lead_contact_ready():
        return "captured"
    if score >= 3 or current_status == "interested":
        return "interested"
    if score >= 1:
        return "warm"
    return current_status or "new"


def build_lead_record(
    session_id: str,
    profile: StudentProfile,
    lead_score: int,
    notes: str = "",
) -> dict[str, Any]:
    program = ", ".join(profile.interested_programs) if profile.interested_programs else None
    return {
        "session_id": session_id,
        "name": profile.name,
        "email": profile.email,
        "phone": profile.phone,
        "interested_program": program,
        "preferred_contact_time": profile.preferred_contact_time,
        "lead_score": lead_score,
        "notes": notes or None,
    }
