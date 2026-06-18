"""Chat LLM providers: Gemini or Groq with automatic model fallback."""

from __future__ import annotations

import config
from src.utils.logger import setup_logger

log = setup_logger("chat.llm")


def active_chat_model() -> str:
    if config.CHAT_MODEL:
        return config.CHAT_MODEL
    if config.CHAT_PROVIDER == "groq":
        return config.GROQ_CHAT_MODEL
    return config.GEMINI_CHAT_MODEL


def _parse_model_list(primary: str, fallbacks_csv: str) -> list[str]:
    models: list[str] = []
    seen: set[str] = set()
    for candidate in [primary, *fallbacks_csv.split(",")]:
        model = candidate.strip()
        if model and model not in seen:
            seen.add(model)
            models.append(model)
    return models


def _groq_chat_models() -> list[str]:
    primary = config.CHAT_MODEL or config.GROQ_CHAT_MODEL
    return _parse_model_list(primary, config.GROQ_CHAT_FALLBACK_MODELS)


def _groq_profile_models() -> list[str]:
    return _parse_model_list(config.GROQ_PROFILE_MODEL, config.GROQ_PROFILE_FALLBACK_MODELS)


def _is_retryable_llm_error(exc: Exception) -> bool:
    name = type(exc).__name__
    if name in {
        "RateLimitError",
        "APIConnectionError",
        "InternalServerError",
        "ServiceUnavailableError",
        "APITimeoutError",
        "APIStatusError",
    }:
        return True

    status_code = getattr(exc, "status_code", None)
    if status_code in {413, 429, 500, 502, 503, 529}:
        return True

    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) in {413, 429, 500, 502, 503, 529}:
        return True

    message = str(exc).lower()
    return (
        "rate limit" in message
        or "quota" in message
        or "overloaded" in message
        or "request too large" in message
        or "tokens per minute" in message
        or "tokens per day" in message
    )


def _gemini_generate(*, system: str, prompt: str, temperature: float | None = None) -> str:
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set")

    from google import genai

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    gen_config: dict = {"system_instruction": system}
    if temperature is not None:
        gen_config["temperature"] = temperature

    response = client.models.generate_content(
        model=config.GEMINI_CHAT_MODEL,
        contents=prompt,
        config=gen_config,
    )
    return response.text or ""


def _gemini_generate_json(prompt: str) -> str:
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set")

    from google import genai

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.GEMINI_PROFILE_MODEL,
        contents=prompt,
        config={
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    )
    return response.text or "{}"


def _groq_client():
    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set")
    from groq import Groq

    return Groq(api_key=config.GROQ_API_KEY)


def _groq_generate(
    *,
    system: str,
    prompt: str,
    model: str,
    temperature: float | None = None,
) -> str:
    client = _groq_client()
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def _groq_generate_json(prompt: str, model: str) -> str:
    client = _groq_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You extract structured data. Reply with valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or "{}"


def _groq_generate_with_fallback(
    *,
    system: str,
    prompt: str,
    models: list[str],
    temperature: float | None = None,
) -> str:
    last_error: Exception | None = None

    for model in models:
        try:
            log.info("Trying Groq chat model: %s", model)
            result = _groq_generate(system=system, prompt=prompt, model=model, temperature=temperature)
            log.info("Groq chat succeeded with model: %s", model)
            return result
        except Exception as exc:
            if not _is_retryable_llm_error(exc):
                log.error("Groq model %s failed with non-retryable error: %s", model, exc)
                raise
            last_error = exc
            log.warning("Groq model %s failed (%s), trying next model...", model, exc)

    log.error("All Groq chat models exhausted: %s", ", ".join(models))

    if last_error is not None:
        raise last_error
    raise ValueError("No Groq chat models configured")


def _groq_generate_json_with_fallback(prompt: str, models: list[str]) -> str:
    last_error: Exception | None = None

    for model in models:
        try:
            log.info("Trying Groq profile model: %s", model)
            result = _groq_generate_json(prompt, model=model)
            log.info("Groq profile succeeded with model: %s", model)
            return result
        except Exception as exc:
            if not _is_retryable_llm_error(exc):
                log.error("Groq profile model %s failed with non-retryable error: %s", model, exc)
                raise
            last_error = exc
            log.warning("Groq profile model %s failed (%s), trying next model...", model, exc)

    log.error("All Groq profile models exhausted: %s", ", ".join(models))

    if last_error is not None:
        raise last_error
    raise ValueError("No Groq profile models configured")


def generate_text(*, system: str, prompt: str, temperature: float | None = None) -> str:
    if config.CHAT_PROVIDER == "groq":
        try:
            return _groq_generate_with_fallback(
                system=system,
                prompt=prompt,
                models=_groq_chat_models(),
                temperature=temperature,
            )
        except Exception as exc:
            if config.CHAT_FALLBACK_TO_GEMINI and config.GEMINI_API_KEY and _is_retryable_llm_error(exc):
                log.warning(
                    "All Groq chat models failed, falling back to Gemini (%s): %s",
                    config.GEMINI_CHAT_MODEL,
                    exc,
                )
                return _gemini_generate(system=system, prompt=prompt, temperature=temperature)
            raise
    return _gemini_generate(system=system, prompt=prompt, temperature=temperature)


def generate_json(prompt: str) -> str:
    if config.CHAT_PROVIDER == "groq":
        try:
            return _groq_generate_json_with_fallback(prompt, _groq_profile_models())
        except Exception as exc:
            if config.CHAT_FALLBACK_TO_GEMINI and config.GEMINI_API_KEY and _is_retryable_llm_error(exc):
                log.warning("All Groq profile models failed, falling back to Gemini: %s", exc)
                return _gemini_generate_json(prompt)
            raise
    return _gemini_generate_json(prompt)
