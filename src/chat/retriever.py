"""Hybrid vector + FTS retrieval from Supabase."""

import re

import config
from src.ingest.embeddings import embed_texts
from src.ingest.supabase_client import get_supabase_client

_COMPARATIVE_SIGNALS = (
    "which",
    "better",
    " vs ",
    " versus ",
    " or ",
    "compare",
    "difference",
    "good for me",
    "should i",
    "recommend",
    "suit me",
    "choose between",
    "help me decide",
)

_FEE_KEYWORDS = ("fee", "fees", "cost", "tuition", "price", "expensive", "afford")
_ADMISSION_KEYWORDS = ("admission", "apply", "eligibility", "entry", "criteria", "requirement")

_SPLIT_PATTERN = re.compile(r"\s+or\s+|\s+vs\.?\s+|\s+versus\s+", re.I)
_LEADING_QUESTION = re.compile(
    r"^(?:which|what|how|should i|help me|is|are|can you|do you)\b(?:\s+\w+){0,12}\s+",
    re.I,
)


def _is_comparative_or_advisory(question: str) -> bool:
    q = question.lower()
    return any(s in q for s in _COMPARATIVE_SIGNALS)


def _clean_comparison_term(segment: str) -> str:
    text = segment.strip().rstrip("?").strip()
    if "?" in text:
        text = text.split("?")[-1].strip()
    text = _LEADING_QUESTION.sub("", text).strip()
    text = re.sub(r"^(?:better|best|good for me)\s+", "", text, flags=re.I).strip()
    return text if len(text) > 1 else ""


def _extract_comparison_terms(question: str) -> list[str]:
    segments = _SPLIT_PATTERN.split(question.strip())
    if len(segments) < 2:
        return []

    terms: list[str] = []
    for segment in segments:
        term = _clean_comparison_term(segment)
        if term:
            terms.append(term)
    return terms


def _topic_suffix(keywords: tuple[str, ...]) -> str:
    return " ".join(keywords)


def expand_queries(question: str) -> list[str]:
    queries = [question]
    q_lower = question.lower()

    if _is_comparative_or_advisory(question):
        terms = _extract_comparison_terms(question)
        for term in terms:
            queries.append(f"{term} program curriculum objectives fees admission")
        if len(terms) >= 2:
            queries.append(f"compare {' and '.join(terms)} programs differences")

    if any(w in q_lower for w in _FEE_KEYWORDS):
        queries.append(f"{question} {_topic_suffix(_FEE_KEYWORDS)}")

    if any(w in q_lower for w in _ADMISSION_KEYWORDS):
        queries.append(f"{question} {_topic_suffix(_ADMISSION_KEYWORDS)}")

    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        key = q.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(q.strip())
    return unique


def search_knowledge(query: str, match_count: int = 8) -> list[dict]:
    embedding = embed_texts([query])[0]
    client = get_supabase_client()
    response = client.rpc(
        "match_university_knowledge_hybrid",
        {
            "query_embedding": embedding,
            "query_text": query,
            "match_count": match_count,
        },
    ).execute()
    return response.data or []


def _chunk_key(row: dict) -> str:
    meta = row.get("meta_data") or {}
    return str(meta.get("chunk_id") or row.get("id") or row.get("content", "")[:80])


def search_knowledge_multi(question: str, match_count: int | None = None) -> list[dict]:
    limit = match_count or config.CHAT_MATCH_COUNT
    queries = expand_queries(question)
    per_query = max(5, limit // len(queries) + 3)

    merged: list[dict] = []
    seen: set[str] = set()

    for query in queries:
        for row in search_knowledge(query, match_count=per_query):
            key = _chunk_key(row)
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)

    merged.sort(
        key=lambda r: float(r.get("combined_score") or r.get("vector_similarity") or 0),
        reverse=True,
    )
    return merged[:limit]
