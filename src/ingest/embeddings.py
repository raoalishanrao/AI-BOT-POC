"""Gemini embedding generation for Supabase vector store."""

import time

import config

_client = None


def _get_client():
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in environment or .env")
        from google import genai

        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def embed_texts(texts: list[str], max_retries: int = 6) -> list[list[float]]:
    if not texts:
        return []

    from google.genai import errors, types

    client = _get_client()
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=config.EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    output_dimensionality=config.EMBEDDING_DIMENSIONS
                ),
            )
            embeddings = []
            for emb in response.embeddings:
                values = list(emb.values)
                if len(values) != config.EMBEDDING_DIMENSIONS:
                    raise ValueError(
                        f"Expected {config.EMBEDDING_DIMENSIONS} dimensions, got {len(values)}"
                    )
                embeddings.append(values)
            return embeddings
        except errors.ClientError as exc:
            last_error = exc
            if getattr(exc, "code", None) == 429 and attempt < max_retries - 1:
                wait = min(60, 5 * (2**attempt))
                time.sleep(wait)
                continue
            raise

    raise last_error  # type: ignore[misc]
