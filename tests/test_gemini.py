"""Gemini configuration and real HTTP requests, with the network kept offline."""

import json
import sys
from urllib.parse import urlsplit
from types import SimpleNamespace

import pytest
import requests

from core import agent, report_writer
from utils import llm
from utils.config import Settings, settings


@pytest.fixture
def gemini_http(monkeypatch):
    """Use real request preparation; replace only the network send."""
    captured = []
    responses = []

    def send(adapter, request, **kwargs):
        if urlsplit(request.url).hostname != "generativelanguage.googleapis.com":
            pytest.fail("The Gemini key must only be sent to Google's API")
        captured.append(request)
        result = responses.pop(0)
        result.request = request
        return result

    def reject_sdk(*args, **kwargs):
        pytest.fail("The hosted app must use the Gemini REST transport")

    if hasattr(llm, "genai"):
        monkeypatch.setattr(llm.genai, "Client", reject_sdk)
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    monkeypatch.setattr(settings, "llm_api_key", "test-gemini-key")
    monkeypatch.setattr(settings, "llm_model", "gemini-3.5-flash-lite")
    return captured, responses


def api_response(status, data):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(data).encode()
    result.headers["Content-Type"] = "application/json"
    return result


def response(text):
    return api_response(200, {
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": text}]},
            "finishReason": "STOP",
        }]
    })


def test_settings_read_gemini_key_from_streamlit_secrets(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(sys.modules, "streamlit", SimpleNamespace(secrets={
        "GEMINI_API_KEY": "test-cloud-key", "GEMINI_MODEL": "gemini-3.5-flash-lite"
    }))
    config = Settings()
    assert config.llm_available
    assert config.llm_api_key == "test-cloud-key"
    assert config.llm_model == "gemini-3.5-flash-lite"


def test_settings_read_gemini_environment_and_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-local-key")
    monkeypatch.setenv("GEMINI_MODEL", "custom-gemini-model")
    config = Settings()
    assert config.llm_api_key == "test-local-key"
    assert config.llm_model == "custom-gemini-model"
    assert "test-local-key" not in repr(config)


def test_old_provider_key_does_not_enable_gemini(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "old-provider-key")
    monkeypatch.setitem(sys.modules, "streamlit", SimpleNamespace(secrets={}))
    assert not Settings().llm_available


def test_text_request_authenticates_only_to_google(gemini_http):
    captured, responses = gemini_http
    responses.append(response("  Gemini report  "))
    assert llm.generate_text("Summarize revenue", system="Be concise", max_tokens=8192) == "Gemini report"
    request = captured[0]
    assert urlsplit(request.url).hostname == "generativelanguage.googleapis.com"
    assert urlsplit(request.url).path == "/v1beta/models/gemini-3.5-flash-lite:generateContent"
    assert request.headers["x-goog-api-key"] == "test-gemini-key"
    assert "test-gemini-key" not in str(request.url)
    payload = json.loads(request.body)
    assert payload["contents"][0]["parts"][0]["text"] == "Summarize revenue"
    assert payload["systemInstruction"]["parts"][0]["text"] == "Be concise"
    assert payload["generationConfig"]["maxOutputTokens"] == 8192
    assert "temperature" not in payload["generationConfig"]


def test_json_request_uses_json_output_mode(gemini_http):
    captured, responses = gemini_http
    responses.append(response('{"questions": [{"question": "Which region?"}]}'))
    assert llm.generate_json("Ask a question") == {"questions": [{"question": "Which region?"}]}
    payload = json.loads(captured[0].body)
    assert payload["generationConfig"]["responseMimeType"] == "application/json"


def test_invalid_key_becomes_llm_error_without_retrying(gemini_http):
    captured, responses = gemini_http
    responses.append(api_response(400, {"error": {
        "code": 400, "status": "INVALID_ARGUMENT", "message": "API key not valid"
    }}))
    with pytest.raises(llm.LLMError, match="Gemini"):
        llm.generate_text("Report")
    assert len(captured) == 1


def test_temporary_api_failure_is_retried(gemini_http):
    captured, responses = gemini_http
    responses.extend([
        api_response(503, {"error": {"code": 503, "message": "Unavailable"}}),
        response("Recovered report"),
    ])
    assert llm.generate_text("Report") == "Recovered report"
    assert len(captured) == 2


def test_empty_response_becomes_llm_error(gemini_http):
    _, responses = gemini_http
    responses.append(api_response(200, {"candidates": []}))
    with pytest.raises(llm.LLMError, match="empty|text"):
        llm.generate_text("Report")


def test_json_array_becomes_llm_error(gemini_http):
    _, responses = gemini_http
    responses.extend([response("[]"), response("[]")])
    with pytest.raises(llm.LLMError, match="JSON object"):
        llm.generate_json("Plan")


def test_planner_identifies_gemini(gemini_http):
    _, responses = gemini_http
    responses.append(response(json.dumps({
        "problem_type": "regression", "target_column": "revenue",
        "feature_columns": ["spend"], "analyses_to_run": ["correlation_analysis"],
        "clarifying_notes": "", "reasoning": "Predict revenue from spend."
    })))
    profile = {
        "columns": {"revenue": {"dtype": "numeric"}, "spend": {"dtype": "numeric"}},
        "numeric_columns": ["revenue", "spend"], "has_datetime": False,
    }
    plan = agent.run_planning_agent(profile, "What predicts revenue?", {})
    assert plan["planner"] == "gemini"
    assert plan["target_column"] == "revenue"


def test_report_identifies_gemini(gemini_http):
    _, responses = gemini_http
    responses.append(response("## Executive Summary\nRevenue increased."))
    report = report_writer.generate_report({"question": "How is revenue?"})
    assert report == {
        "markdown": "## Executive Summary\nRevenue increased.", "source": "gemini"
    }


def test_sidebar_renders_with_a_gemini_key(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(settings, "llm_api_key", "test-gemini-key")
    monkeypatch.setattr(settings, "db_path", tmp_path / "sessions.db")
    app = AppTest.from_file("app.py").run()
    assert not app.exception
    assert "Gemini" in app.success[0].value
