"""Clean model output for chat display."""

import re

_CITATION_RE = re.compile(r"\[\d+(?:,\s*\d+)*\]")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_BULLET_RE = re.compile(r"^\*\s+", re.M)

FORMAT_INSTRUCTION = """
Formatting rules (strict):
- Plain text only. Do NOT use markdown: no **, no *, no #, no backticks.
- Use the bullet character • for list items.
- Do NOT add citation numbers or source references like [1] or [6].
- Keep paragraphs short and readable in a chat window.
- Keep replies concise: usually 2–6 sentences unless the student asks for a detailed comparison.
- NEVER paste admission, fees, or scholarship URLs — the chat UI shows those as buttons.
- NEVER list or recap the student's name, email, or phone unless they explicitly ask you to confirm them.
- Do NOT start replies with "Hello [name]" every time — use their name sparingly or not at all.
- Do NOT say "To recap", "I have your details", or repeat program recommendations unless the topic changed.
"""


def clean_reply(text: str) -> str:
    if not text:
        return text

    cleaned = _CITATION_RE.sub("", text)
    cleaned = _BOLD_RE.sub(r"\1", cleaned)
    cleaned = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", cleaned)
    cleaned = _BULLET_RE.sub("• ", cleaned)
    cleaned = re.sub(r"^-\s+", "• ", cleaned, flags=re.M)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\s+\.", ".", cleaned)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return _strip_recap_boilerplate(cleaned.strip())


_RECAP_LINE = re.compile(
    r"^(?:to recap|i have your (?:contact )?details|name:|email:|phone:|interested programs?:|"
    r"for your reference|if you.?re ready to move forward|which program are you most interested).*$",
    re.I | re.M,
)
_URL_LINE = re.compile(r"^https?://\S+\s*$", re.M)


def _strip_recap_boilerplate(text: str) -> str:
    lines = text.split("\n")
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if _RECAP_LINE.match(stripped) or _URL_LINE.match(stripped):
            continue
        kept.append(line)
    result = "\n".join(kept).strip()
    return re.sub(r"\n{3,}", "\n\n", result)
