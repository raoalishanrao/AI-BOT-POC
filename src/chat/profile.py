"""Student profile for counseling sessions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

PROFILE_FIELDS = (
    "name",
    "age",
    "qualification",
    "academic_background",
    "grade_cgpa",
    "subjects",
    "career_goals",
    "interests",
    "preferred_industry",
    "study_mode",
    "budget",
    "preferred_intake",
    "interested_programs",
    "email",
    "phone",
    "preferred_contact_time",
)


@dataclass
class StudentProfile:
    name: str | None = None
    age: str | None = None
    qualification: str | None = None
    academic_background: str | None = None
    grade_cgpa: str | None = None
    subjects: list[str] = field(default_factory=list)
    career_goals: str | None = None
    interests: list[str] = field(default_factory=list)
    preferred_industry: str | None = None
    study_mode: str | None = None
    budget: str | None = None
    preferred_intake: str | None = None
    interested_programs: list[str] = field(default_factory=list)
    email: str | None = None
    phone: str | None = None
    preferred_contact_time: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> StudentProfile:
        if not data:
            return cls()
        kwargs: dict[str, Any] = {}
        for key in PROFILE_FIELDS:
            value = data.get(key)
            if value is None:
                continue
            if key in {"subjects", "interests", "interested_programs"} and isinstance(value, str):
                kwargs[key] = [value]
            else:
                kwargs[key] = value
        return cls(**kwargs)

    def merge(self, updates: dict[str, Any]) -> StudentProfile:
        data = self.to_dict()
        for key, value in updates.items():
            if key not in PROFILE_FIELDS or value in (None, "", []):
                continue
            if key in {"subjects", "interests", "interested_programs"}:
                existing = set(data.get(key) or [])
                if isinstance(value, str):
                    existing.add(value.strip())
                else:
                    existing.update(str(v).strip() for v in value if str(v).strip())
                data[key] = sorted(existing)
            else:
                data[key] = str(value).strip() if not isinstance(value, str) else value.strip()
        return StudentProfile.from_dict(data)

    def contact_fields_missing(self) -> list[str]:
        missing: list[str] = []
        if not self.name:
            missing.append("name")
        if not self.email and not self.phone:
            missing.append("email or WhatsApp number")
        return missing

    def has_contact_info(self) -> bool:
        return bool(self.name and (self.email or self.phone))

    def profiling_fields_missing(self) -> list[str]:
        missing: list[str] = []
        if not self.qualification and not self.academic_background:
            missing.append("qualification or academic background")
        if not self.career_goals and not self.interests:
            missing.append("career goals or areas of interest")
        if not self.grade_cgpa:
            missing.append("grades/CGPA (if known)")
        return missing

    def is_ready_for_recommendation(self) -> bool:
        has_background = bool(self.qualification or self.academic_background)
        has_goals = bool(self.career_goals or self.interests)
        return has_background and has_goals

    def lead_contact_ready(self) -> bool:
        """Enough contact info to sync a lead record (does not require program choice)."""
        return bool(self.name and (self.email or self.phone))

    def to_context(self) -> str:
        lines: list[str] = []
        for key in PROFILE_FIELDS:
            value = getattr(self, key)
            if not value:
                continue
            label = key.replace("_", " ").title()
            if isinstance(value, list):
                lines.append(f"- {label}: {', '.join(value)}")
            else:
                lines.append(f"- {label}: {value}")
        return "\n".join(lines) if lines else "(No profile collected yet.)"


def profile_updates_from_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if k in PROFILE_FIELDS and v not in (None, "", [])}
