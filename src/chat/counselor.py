"""AI counseling agent with profiling, RAG, and lead generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import config
from src.chat.leads import (
    detect_lead_score,
    extract_contact_hints,
    qualify_lead_status,
)
from src.chat.profile import StudentProfile, profile_updates_from_json
from src.chat.retriever import search_knowledge_multi
from src.chat.session_store import (
    SessionState,
    append_message,
    create_session,
    get_session,
    save_session,
)
from src.chat.users import sync_session_user_and_lead
from src.chat.formatting import FORMAT_INSTRUCTION, clean_reply
from src.chat.llm import generate_json, generate_text
from src.utils.logger import setup_logger

log = setup_logger("chat.counselor")

SYSTEM_PROMPT = f"""You are the AI Education Counselor for Iqra University Chak Shahzad Campus, Islamabad.

Your role matches a virtual admissions counselor:
1. Welcome prospective students warmly.
2. Learn their background and goals through natural conversation — be curious, not interrogative.
3. Recommend suitable programs with clear reasoning when you have enough context.
4. Answer questions about programs, fees, scholarships, admissions, and policies using ONLY the provided knowledge context.
5. Guide interested students toward next steps (apply, inquiry, speak with admissions).
6. Remember what the student already shared — never ask for the same information twice.
7. Contact details (name, email, phone) must feel optional and human — never like a registration form or CRM bot.
   - When you need contact info, ask for ALL missing items together in one warm sentence (not one-by-one).
   - Lead with value: e.g. "I can have admissions email you the fee sheet" or "they can call you about scholarships" — not "so I can assist you better."
   - Help first. Do not ask for contact on every message. At most one light invite every few turns.
   - If they already shared name, do not ask for name again. Never use robotic repeated phrases.
8. You may give useful answers even before all contact details are collected.

Counseling rules:
- Base factual claims on the knowledge context. Compare programs when asked.
- Explain WHY a program fits the student's background, interests, and goals.
- Include fees (PKR) and entry criteria when relevant and available in context.
- Answer the student's latest question directly first — do not append unrelated recaps.
- Contact info is stored in the system — never repeat it back unless they ask to confirm.
- Program recommendations belong when interests change or they ask — not on every message.
- Mention apply/admissions next steps only when they show interest, not after every reply.

Contact: {config.ADMISSIONS_EMAIL}, {config.ADMISSIONS_PHONE}

Keep responses concise, structured, and friendly.
{FORMAT_INSTRUCTION}"""


@dataclass
class CounselorResponse:
    reply: str
    session_id: str
    stage: str
    lead_status: str
    profile: dict[str, Any]
    recommended_programs: list[str]
    ctas: list[dict[str, str]]


def _build_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(No matching context retrieved.)"

    parts = []
    for i, row in enumerate(chunks, 1):
        meta = row.get("meta_data") or {}
        title = meta.get("title") or meta.get("slug") or "Source"
        url = meta.get("source_url", "")
        header = f"[{i}] {title}"
        if url:
            header += f" ({url})"
        parts.append(f"{header}\n{row.get('content', '')}")
    return "\n\n---\n\n".join(parts)


def _history_text(messages: list[dict[str, str]], limit: int | None = None) -> str:
    limit = limit or config.CHAT_HISTORY_LIMIT
    recent = messages[-limit:]
    lines = []
    for msg in recent:
        role = "Student" if msg["role"] == "user" else "Counselor"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines) if lines else "(No prior messages.)"


def _retrieval_query(message: str, profile: StudentProfile) -> str:
    parts = [message]
    if profile.career_goals:
        parts.append(profile.career_goals)
    if profile.interests:
        parts.extend(profile.interests)
    if profile.interested_programs:
        parts.extend(profile.interested_programs)
    if profile.qualification:
        parts.append(profile.qualification)
    return " ".join(parts)


def _normalize_program_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name).strip().rstrip(":.")
    cleaned = re.sub(r"\s+is Rs.*$", "", cleaned, flags=re.I)
    return cleaned[:80]


def _extract_program_names(reply: str) -> list[str]:
    found: set[str] = set()
    for match in re.findall(r"\*\*((?:BS|Bachelor|Doctor|ADP|Diploma)[^*]+)\*\*", reply, re.I):
        found.add(_normalize_program_name(match))
    for match in re.findall(r"\b(BS in [A-Za-z][\w\s&-]{3,45})\b", reply):
        found.add(_normalize_program_name(match))
    return sorted(n for n in found if len(n) > 8)[:5]


def _extract_profile_hints(message: str) -> dict[str, Any]:
    hints = extract_contact_hints(message)
    name_match = re.search(
        r"(?:my name is|i am|i'm|this is)\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,2})",
        message,
        re.I,
    )
    if name_match:
        hints["name"] = name_match.group(1).strip().title()
    elif re.fullmatch(r"[A-Za-z]+(?:\s+[A-Za-z]+){1,2}", message.strip()):
        hints["name"] = message.strip().title()
    program_match = re.search(
        r"(?:apply for|interested in|want to (?:study|pursue))\s+(.+?)(?:\.|$)",
        message,
        re.I,
    )
    if program_match:
        hints["interested_programs"] = [program_match.group(1).strip()[:80]]
    return hints


def _update_stage(state: SessionState, message: str = "") -> None:
    profile = state.profile
    if _is_farewell(message):
        state.stage = "completed"
    elif not profile.has_contact_info():
        state.stage = "contact_collection"
    elif detect_lead_score(message) >= 3 and profile.has_contact_info():
        state.stage = "lead_capture"
    elif profile.is_ready_for_recommendation() and state.recommended_programs:
        state.stage = "qa"
    elif profile.is_ready_for_recommendation():
        state.stage = "recommending"
    elif state.messages:
        state.stage = "profiling"
    else:
        state.stage = "introduction"


def _is_farewell(message: str) -> bool:
    return bool(re.search(r"\b(bye|goodbye|thanks|thank you|that's all|done for now)\b", message, re.I))


def _is_simple_factual_query(message: str) -> bool:
    q = message.lower().strip()
    if len(q) > 120:
        return False
    return bool(
        re.search(
            r"\b(where|location|located|address|campus|open|hours|phone number of|contact)\b",
            q,
        )
    )


def _profile_for_prompt(profile: StudentProfile) -> str:
    if profile.has_contact_info():
        parts = ["(Contact on file — do NOT mention or list name, email, or phone in your reply.)"]
    else:
        parts = [profile.to_context()]

    extras: list[str] = []
    if profile.qualification or profile.academic_background:
        extras.append(f"Qualification: {profile.qualification or profile.academic_background}")
    if profile.grade_cgpa:
        extras.append(f"Grades: {profile.grade_cgpa}")
    if profile.career_goals:
        extras.append(f"Career goals: {profile.career_goals}")
    if profile.interests:
        extras.append(f"Interests: {', '.join(profile.interests)}")
    if profile.interested_programs:
        extras.append(f"Programs discussed: {', '.join(profile.interested_programs[:5])}")
    if extras:
        parts.extend(extras)
    return "\n".join(parts)


def _build_ctas(state: SessionState) -> list[dict[str, str]]:
    if state.lead_status not in {"interested", "captured"}:
        return []
    if state.stage != "lead_capture" and state.lead_status != "captured":
        return []

    ctas = [
        {"label": "Apply Now", "url": config.ADMISSIONS_URL},
        {"label": "View Fees", "url": config.FEES_URL},
        {"label": "Scholarships", "url": config.SCHOLARSHIPS_URL},
        {"label": "Contact Admissions", "url": config.CONTACT_URL},
    ]
    if state.lead_status in {"interested", "warm", "captured"}:
        ctas.insert(0, {"label": "Email Admissions", "url": f"mailto:{config.ADMISSIONS_EMAIL}"})
    return ctas


def _should_nudge_contact(state: SessionState, profile: StudentProfile) -> bool:
    if profile.has_contact_info():
        return False
    if state.contact_nudges >= 2:
        return False
    user_turns = sum(1 for m in state.messages if m["role"] == "user")
    if user_turns < 2:
        return False
    if state.lead_status in {"warm", "interested", "captured"}:
        return True
    return user_turns >= 3 and user_turns % 2 == 0


def _contact_guidance(profile: StudentProfile, should_nudge: bool) -> str:
    missing = profile.contact_fields_missing()
    if not missing:
        return "Contact info is complete. Do not ask for it."
    if not should_nudge:
        return (
            "Do NOT ask for contact details this turn. Focus only on answering and helping. "
            "Be genuinely useful — build trust first."
        )
    fields = " and ".join(missing)
    return (
        f"Still missing: {fields}. In ONE brief, warm sentence, invite them to share "
        f"{'these' if len(missing) > 1 else 'this'} together — optional tone, mention a clear benefit "
        "(fee details, scholarship info, or admissions callback). "
        "Never ask field-by-field. Never say 'assist you better' or 'send you updates'."
    )


def _extract_profile_updates(message: str, profile: StudentProfile, history: str) -> dict[str, Any]:
    prompt = f"""Extract any NEW student profile fields mentioned in the latest message.
Return ONLY valid JSON with keys from this list (omit unknowns):
name, age, qualification, academic_background, grade_cgpa, subjects, career_goals, interests,
preferred_industry, study_mode, budget, preferred_intake, interested_programs, email, phone, preferred_contact_time

Current profile:
{profile.to_context()}

Recent conversation:
{history}

Latest student message:
{message}

JSON:"""

    try:
        raw = generate_json(prompt)
        return profile_updates_from_json(raw)
    except Exception as exc:
        log.warning("Profile extraction failed: %s", exc)
        return extract_contact_hints(message)


def _generate_reply(
    *,
    message: str,
    profile: StudentProfile,
    history: str,
    rag_context: str,
    state: SessionState,
) -> str:
    missing_contact = profile.contact_fields_missing()
    missing_profile = profile.profiling_fields_missing()
    should_nudge = _should_nudge_contact(state, profile)
    simple = _is_simple_factual_query(message)
    stage_guidance = {
        "introduction": (
            "Welcome warmly. Answer what they asked. "
            "Do not ask for email or phone yet — just be helpful."
        ),
        "contact_collection": _contact_guidance(profile, should_nudge),
        "profiling": (
            f"Learn about their background. Missing: "
            f"{', '.join(missing_profile) if missing_profile else 'none — move to recommendations'}."
        ),
        "recommending": (
            "Recommend 1–3 best-fit programs with reasoning, entry requirements, fees if known."
        ),
        "qa": (
            "Answer ONLY what they asked. Do not recap contact info, past recommendations, or links. "
            "2–5 sentences unless they want a comparison."
        ),
        "lead_capture": (
            "They want to apply or take next steps. One short paragraph: how to apply, offer admissions callback. "
            "No contact recap."
        ),
        "completed": "Brief friendly sign-off. Summarize top program pick only if obvious.",
    }
    if simple:
        stage_guidance[state.stage] = (
            "Simple factual question. Reply in 1–3 sentences. "
            "No greetings, no name, no contact recap, no program recap, no links."
        )

    prompt = f"""Session stage: {state.stage}
Lead status: {state.lead_status}
Contact nudges so far: {state.contact_nudges}
Recommended so far: {', '.join(state.recommended_programs) if state.recommended_programs else 'none'}

Student profile (internal — do not echo contact fields):
{_profile_for_prompt(profile)}

Conversation history:
{history}

Knowledge context (use for facts):
{rag_context}

Stage guidance: {stage_guidance.get(state.stage, '')}

Latest student message:
{message}

Respond as the counselor."""

    return generate_text(system=SYSTEM_PROMPT, prompt=prompt)


def _greeting() -> str:
    return (
        "Hello! I'm your AI Education Counselor for Iqra University, Chak Shahzad Campus. "
        "Whether you're exploring programs, fees, scholarships, or just figuring out what fits you — "
        "I'm happy to help.\n\n"
        "What would you like to know today?"
    )


def start_session(device_info: dict[str, Any] | None = None) -> CounselorResponse:
    state = create_session(device_info=device_info)
    greeting = _greeting()
    append_message(state, "assistant", greeting)
    save_session(state)
    return CounselorResponse(
        reply=greeting,
        session_id=state.session_id,
        stage=state.stage,
        lead_status=state.lead_status,
        profile=state.profile.to_dict(),
        recommended_programs=[],
        ctas=_build_ctas(state),
    )


def chat(session_id: str, message: str, match_count: int | None = None) -> CounselorResponse:
    state = get_session(session_id)
    if not state:
        raise ValueError(f"Unknown session: {session_id}")

    message = message.strip()
    if not message:
        raise ValueError("Message cannot be empty")

    history = _history_text(state.messages)
    append_message(state, "user", message)

    updates = _extract_profile_updates(message, state.profile, history)
    hints = _extract_profile_hints(message)
    merged_updates = {**updates, **{k: v for k, v in hints.items() if k not in updates or not updates[k]}}
    if merged_updates:
        state.profile = state.profile.merge(merged_updates)

    sync_session_user_and_lead(state)

    state.lead_score += detect_lead_score(message)
    state.lead_status = qualify_lead_status(state.profile, message, state.lead_status)
    _update_stage(state, message)

    query = _retrieval_query(message, state.profile)
    chunks = search_knowledge_multi(query, match_count=match_count)
    rag_context = _build_context(chunks)

    raw_reply = _generate_reply(
        message=message,
        profile=state.profile,
        history=_history_text(state.messages),
        rag_context=rag_context,
        state=state,
    )

    if _should_nudge_contact(state, state.profile) and not state.profile.has_contact_info():
        state.contact_nudges += 1

    programs = _extract_program_names(raw_reply)
    reply = clean_reply(raw_reply)
    if programs:
        merged = set(state.recommended_programs)
        merged.update(programs)
        state.recommended_programs = sorted(merged)[:8]
        state.profile = state.profile.merge({"interested_programs": state.recommended_programs})

    append_message(state, "assistant", reply)
    _update_stage(state, message)

    if state.profile.lead_contact_ready() and state.lead_status != "captured":
        state.lead_status = "captured"

    sync_session_user_and_lead(state)
    save_session(state)

    return CounselorResponse(
        reply=reply,
        session_id=state.session_id,
        stage=state.stage,
        lead_status=state.lead_status,
        profile=state.profile.to_dict(),
        recommended_programs=state.recommended_programs,
        ctas=_build_ctas(state),
    )


def ask(question: str, match_count: int | None = None) -> str:
    """Stateless one-shot Q&A (backward compatible)."""
    from src.chat.bot import ask as stateless_ask

    return stateless_ask(question, match_count=match_count)
