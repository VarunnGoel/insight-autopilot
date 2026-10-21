"""Gemini generation shared by the planner and report writer.

Uses the official Gemini REST API through requests. API errors become ``LLMError`` so callers
can continue using their existing offline fallbacks.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests

from utils.config import settings


class LLMError(RuntimeError):
    """Raised when Gemini cannot return a usable response."""


def llm_is_available() -> bool:
    """True when a Gemini API key is configured."""
    return settings.llm_available


def _generate_content(
    prompt: str,
    system: Optional[str],
    max_tokens: int,
    *,
    json_output: bool = False,
) -> str:
    if not llm_is_available():
        raise LLMError("No GEMINI_API_KEY configured in Streamlit Secrets or .env.")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{quote(settings.llm_model, safe='')}:generateContent"
    )
    generation_config = {
        "maxOutputTokens": max_tokens,
        "thinkingConfig": {"thinkingLevel": "low"},
    }
    if json_output:
        generation_config["responseMimeType"] = "application/json"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    for attempt in range(1, settings.llm_max_retries + 1):
        try:
            response = requests.post(
                url,
                json=payload,
                headers={"x-goog-api-key": settings.llm_api_key},
                timeout=settings.llm_timeout_seconds,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            if attempt == settings.llm_max_retries:
                raise LLMError("Gemini API request failed after retries.") from exc
            time.sleep(min(2 ** attempt, 15))
            continue

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError("Gemini returned an invalid JSON response.") from exc
        if not isinstance(data, dict):
            raise LLMError("Gemini returned an unexpected response.")
        if response.status_code != 200:
            if response.status_code in (429, 500, 502, 503, 504) and attempt < settings.llm_max_retries:
                time.sleep(min(2 ** attempt, 15))
                continue
            message = str(data.get("error", {}).get("message", "Request rejected"))
            message = message.replace(settings.llm_api_key, "[key hidden]")
            raise LLMError(f"Gemini API request failed (HTTP {response.status_code}): {message}")

        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMError("Gemini returned an empty text response.")
        candidate = candidates[0]
        if candidate.get("finishReason") == "MAX_TOKENS":
            raise LLMError("Gemini reached the output token limit; increase max_tokens.")
        parts = candidate.get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
        if not text.strip():
            raise LLMError("Gemini returned an empty text response.")
        return text.strip()
    raise LLMError("Gemini API request failed after retries.")


def generate_text(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.4,
    max_tokens: int = 8192,
) -> str:
    """Return free-form text from Gemini, or raise ``LLMError``.

    ``temperature`` is retained for existing callers. Gemini 3 models use their
    default sampling settings, as recommended by Google.
    """
    return _generate_content(prompt, system, max_tokens)


def _strip_code_fences(raw: str) -> str:
    """Remove Markdown fences when present."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def _parse_json_object(raw: str) -> Dict[str, Any]:
    """Accept a JSON object, including one enclosed in Markdown fences."""
    parsed = json.loads(_strip_code_fences(raw))
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def generate_json(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """Request JSON output and retry once if the result is not a JSON object."""
    for attempt in range(2):
        instruction = prompt
        if attempt:
            instruction += "\n\nReturn ONLY a single valid JSON object. No prose, no fences."
        raw = _generate_content(instruction, system, max_tokens, json_output=True)
        try:
            return _parse_json_object(raw)
        except ValueError:
            if attempt:
                raise LLMError("Gemini did not return a valid JSON object.") from None
    raise LLMError("Gemini did not return a valid JSON object.")
