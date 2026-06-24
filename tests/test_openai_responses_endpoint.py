"""Tests for OpenAI Responses endpoint."""

import json
import pytest


def test_responses_final_answer(app_final_answer):
    """POST /v1/responses should return final answer."""
    client = app_final_answer
    response = client.post("/v1/responses", json={
        "model": "qwable-fast",
        "input": "Hello",
        "stream": False,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "response"
    assert data["status"] == "completed"
    assert len(data["output"]) == 1
    assert data["output"][0]["type"] == "output_text"
    assert data["output"][0]["text"] == "Hello world"


def test_responses_tool_call(app_tool_call):
    """POST /v1/responses should return tool call."""
    client = app_tool_call
    response = client.post("/v1/responses", json={
        "model": "qwable-fast",
        "input": "Read file",
        "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        "stream": False,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["output"][0]["type"] == "function_call"
    assert isinstance(data["output"][0]["arguments"], str)
    assert json.loads(data["output"][0]["arguments"]) == {"path": "test.txt"}


def test_responses_debug_includes_trace_when_requested(app_heavy_debug_answer):
    """POST /v1/responses should include fusion trace only when debug metadata is true."""
    client = app_heavy_debug_answer
    response = client.post("/v1/responses", json={
        "model": "qwable-heavy",
        "input": "Analyze",
        "metadata": {"debug": True},
        "stream": False,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["output"][0]["text"] == "heavy answer"
    assert data["debug"]["profile"] == "heavy-agent"
    assert data["debug"]["heavy_backend"] == "ds4"
    assert data["debug"]["fallback"] is None
    assert data["debug"]["resource_guard"] is True
    assert "175.0GB exceeds limit" in data["debug"]["reason"]
    assert "heavy_resource_guard" in data["debug"]["rationale_summary"]


def test_responses_debug_omitted_without_debug_metadata(app_heavy_debug_answer):
    """POST /v1/responses should not expose trace unless debug metadata is true."""
    client = app_heavy_debug_answer
    response = client.post("/v1/responses", json={
        "model": "qwable-heavy",
        "input": "Analyze",
        "stream": False,
    })
    assert response.status_code == 200
    data = response.json()
    assert "debug" not in data
