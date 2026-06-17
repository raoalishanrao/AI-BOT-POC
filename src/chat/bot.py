"""Chatbot with RAG context from Supabase (stateless Q&A)."""

from src.chat.formatting import FORMAT_INSTRUCTION, clean_reply
from src.chat.llm import generate_text
from src.chat.retriever import search_knowledge_multi

SYSTEM_PROMPT = """You are an intelligent admissions advisor for Iqra University Chak Shahzad Campus, Islamabad.

Your job is to help prospective students understand programs, fees, admissions, and policies — and to guide them toward a good fit when they ask for advice.

Rules:
1. Base answers on the provided context. Synthesize and compare when the context describes multiple programs or topics.
2. For advisory or comparison questions (e.g. "which is better", "X or Y"):
   - Compare using focus, curriculum emphasis, outcomes, fees, and entry criteria found in the context
   - Map the user's implied interests to the option that fits — without inventing facts not in context
   - Ask 1–2 brief clarifying questions when that would help personalize the recommendation
3. Include relevant fees (PKR) and entry criteria when they appear in the context.
4. Only say information is unavailable if the context truly lacks anything useful for the question — do NOT refuse comparative questions when relevant descriptions are present.
5. Be warm, clear, and structured (short bullets when comparing options).
""" + FORMAT_INSTRUCTION + """

Contact if needed: admissions@iqrauni.edu.pk, 051 9247407 / 051 8357378."""


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


def ask(question: str, match_count: int | None = None) -> str:
    chunks = search_knowledge_multi(question, match_count=match_count)
    context = _build_context(chunks)

    prompt = f"""Context from Iqra University website (use this to answer, including comparisons and recommendations):

{context}

---

User question: {question}

Answer helpfully. If comparing programs, structure the comparison clearly."""

    raw = generate_text(system=SYSTEM_PROMPT, prompt=prompt)
    return clean_reply(raw)
