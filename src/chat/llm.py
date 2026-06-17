"""Chat LLM providers: Gemini or Groq."""

from __future__ import annotations

import json

import config
from src.utils.logger import setup_logger

log = setup_logger("chat.llm")


def active_chat_model() -> str:
    if config.CHAT_MODEL:
        return config.CHAT_MODEL
    if config.CHAT_PROVIDER == "groq":
        return config.GROQ_CHAT_MODEL
    return config.GEMINI_CHAT_MODEL


def _gemini_generate(*, system: str, prompt: str, temperature: float | None = None) -> str:
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set")

    from google import genai

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    gen_config: dict = {"system_instruction": system}
    if temperature is not None:
        gen_config["temperature"] = temperature

    response = client.models.generate_content(
        model=active_chat_model(),
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
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    client = _groq_client()
    kwargs: dict = {
        "model": model or active_chat_model(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def _groq_generate_json(prompt: str) -> str:
    client = _groq_client()
    response = client.chat.completions.create(
        model=config.GROQ_PROFILE_MODEL,
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


def generate_text(*, system: str, prompt: str, temperature: float | None = None) -> str:
    if config.CHAT_PROVIDER == "groq":
        return _groq_generate(system=system, prompt=prompt, temperature=temperature)
    return _gemini_generate(system=system, prompt=prompt, temperature=temperature)


def generate_json(prompt: str) -> str:
    if config.CHAT_PROVIDER == "groq":
        return _groq_generate_json(prompt)
    return _gemini_generate_json(prompt)
