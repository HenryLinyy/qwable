"""Tests for Anthropic Messages streaming."""

import pytest


def test_anthropic_streaming(app_anthropic_final_answer):
    """POST /v1/messages with stream=True should return SSE."""
    client = app_anthropic_final_answer
    response = client.post("/v1/messages", json={
        "model": "claude-qwable-fast",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }, headers={
        "x-api-key": "local",
        "anthropic-version": "2023-06-01",
    })
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "event: message_start" in body
    assert "event: content_block_delta" in body
    assert "event: message_stop" in body


def test_anthropic_streaming_tool_use(app_anthropic_tool_use):
    """POST /v1/messages with stream=True and tool_use."""
    client = app_anthropic_tool_use
    response = client.post("/v1/messages", json={
        "model": "claude-qwable-fast",
        "max_tokens": 1024,
        "tools": [{"name": "Bash", "input_schema": {}}],
        "messages": [{"role": "user", "content": "List directory"}],
        "stream": True,
    }, headers={
        "x-api-key": "local",
        "anthropic-version": "2023-06-01",
    })
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "event: content_block_start" in body
    assert '"type": "tool_use"' in body
    assert "event: message_stop" in body
