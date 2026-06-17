"""Text chunking utilities for RAG."""

import re


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if end < len(text):
            break_at = chunk.rfind("\n\n")
            if break_at > chunk_size // 2:
                chunk = chunk[:break_at]
                end = start + break_at

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < len(text) else len(text)

    return chunks
