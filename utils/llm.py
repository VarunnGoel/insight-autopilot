"""Anthropic-compatible LLM API client.

We talk to the LLM using the standard Anthropic Messages API format.
Keeps things lightweight with ``requests`` (no SDK needed).

Public helpers:
    - ``generate_text(...)``    -> free-form text (used by the report writer)
    - ``generate_json(...)``    -> parsed JSON dict (used by the planning agent)
    - ``llm_is_available()``    -> bool, so callers can pick a fallback path
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from utils.config import settings
from utils.logger import get_logger

log = get_logger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM cannot return a usable response."""


def llm_is_available() -> bool:
    """True when a Kintio API key is configured."""
    return settings.llm_available


def _parse_sse(raw: str) -> Dict[str, Any]:
    """Parse Kintio's SSE stream into a response dict matching the non-streaming shape.

    Kintio always returns SSE even without ``stream: true``, so we assemble
    the ``message_start`` content blocks + ``content_block_delta`` text pieces.
    """
    msg: Optional[Dict[str, Any]] = None
    content_blocks: List[Dict[str, Any]] = []
    current_block: Optional[Dict[str, Any]] = None

    for line in raw.splitlines():
        if line.startswith("data: "):
            data = json.loads(line[6:])
            t = data.get("type")
            if t == "message_start":
                msg = data["message"]
                content_blocks = msg.get("content", [])
            elif t == "content_block_start":
                current_block = data["content_block"]
            elif t == "content_block_delta":
                if current_block is not None:
                    current_block.setdefault("text", "")
                    current_block["text"] += data["delta"].get("text", "")
            elif t == "content_block_stop" and current_block is not None:
                content_blocks.append(current_block)
                current_block = None
            elif t == "message_delta":
                if msg is not None:
                    msg["stop_reason"] = data["delta"].get("stop_reason")
                    msg["stop_sequence"] = data["delta"].get("stop_sequence")
                    msg["usage"] = data.get("usage", {})

    if msg is None:
        raise LLMError("No message_start event found in SSE stream")
    msg["content"] = content_blocks
    return msg


def _post(payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST to the Messages API, handling JSON or SSE responses."""
    import requests

    url = f"{settings.llm_base_url}/v1/messages"
    headers = {
        "x-api-key": settings.llm_api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    last_error: Optional[Exception] = None
    for attempt in range(1, settings.llm_max_retries + 1):
        try:
            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=settings.llm_timeout_seconds,
            )
            if resp.status_code == 200:
                if "text/event-stream" in resp.headers.get("Content-Type", "") or "data:" in resp.text:
                    return _parse_sse(resp.text)
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = min(2**attempt, 15)
                log.warning(
                    "API returned %s (attempt %s/%s); retrying in %ss",
                    resp.status_code,
                    attempt,
                    settings.llm_max_retries,
                    wait,
                )
                time.sleep(wait)
                last_error = LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                continue
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        except Exception as exc:
            last_error = exc
            log.warning("API request failed (attempt %s): %s", attempt, exc)
            time.sleep(min(2**attempt, 15))
    raise LLMError(f"API request failed after retries: {last_error}")


def _extract_text(response: Dict[str, Any]) -> str:
    """Pull the text out of an Anthropic Messages API response."""
    try:
        return "".join(
            block["text"] for block in response["content"] if block["type"] == "text"
        ).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected API response shape: {exc}") from exc


def generate_text(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.4,
    max_tokens: int = 8192,
) -> str:
    """Return free-form text from the LLM.

    Raises LLMError if no key is configured or the call fails; callers that want
    a fallback should check ``llm_is_available()`` first.
    """
    if not llm_is_available():
        raise LLMError("No LLM_API_KEY configured.")

    payload: Dict[str, Any] = {
        "model": settings.llm_model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system

    response = _post(payload)
    return _extract_text(response)


def _strip_code_fences(raw: str) -> str:
    """Remove ```json ... ``` fences that models often wrap JSON in."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def _first_json_object(text: str) -> str:
    """Best-effort extraction of the first balanced {...} block from text."""
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def generate_json(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """Return a parsed JSON object from the LLM.

    Asks the LLM for JSON output, strips code fences, and parses. Retries once
    with a stricter instruction if the first parse fails.
    """
    if not llm_is_available():
        raise LLMError("No LLM_API_KEY configured.")

    raw = generate_text(
        prompt,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    for candidate in (_strip_code_fences(raw), _first_json_object(raw)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # One stricter retry.
    strict = generate_text(
        prompt + "\n\nReturn ONLY a single valid JSON object. No prose, no fences.",
        system=system,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    try:
        return json.loads(_first_json_object(_strip_code_fences(strict)))
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"Could not parse JSON from LLM output: {exc}\nRaw: {raw[:300]}"
        )
