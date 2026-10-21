"""Gemini generation shared by the planner and report writer.

Uses the official Google GenAI SDK. API errors become ``LLMError`` so callers
can continue using their existing offline fallbacks.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import httpx
from google import genai
from google.genai import errors, types

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

    try:
        with genai.Client(
            api_key=settings.llm_api_key,
            http_options=types.HttpOptions(
                timeout=settings.llm_timeout_seconds * 1000,
                retry_options=types.HttpRetryOptions(
                    attempts=settings.llm_max_retries,
                    http_status_codes=[429, 500, 502, 503, 504],
                ),
            ),
        ) as client:
            response = client.models.generate_content(
                model=settings.llm_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json" if json_output else None,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            )
    except (errors.APIError, httpx.HTTPError, ValueError) as exc:
        raise LLMError(f"Gemini API request failed: {exc}") from exc

    if response.candidates and response.candidates[0].finish_reason == types.FinishReason.MAX_TOKENS:
        raise LLMError("Gemini reached the output token limit; increase max_tokens.")
    text = response.text
    if not text or not text.strip():
        raise LLMError("Gemini returned an empty text response.")
    return text.strip()


def generate_text(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.4,
    max_tokens: int = 8192,
) -> str:
    """Return free-form text from Gemini, or raise ``LLMError``.

    ``temperature`` is retained for existing callers. Gemini 3.8 uses its model
    default; custom sampling parameters are deprecated for that model.
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
