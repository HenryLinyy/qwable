"""Tests for Anthropic Messages endpoint."""


def test_anthropic_final_answer(app_anthropic_final_answer):
    """POST /v1/messages should return final answer."""
    client = app_anthropic_final_answer
    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-qwable-fast",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
        headers={
            "x-api-key": "local",
            "anthropic-version": "2023-06-01",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "message"
    assert len(data["content"]) == 1
    assert data["content"][0]["type"] == "text"
    assert data["content"][0]["text"] == "Hello world"


def test_anthropic_tool_use(app_anthropic_tool_use):
    """POST /v1/messages should return tool_use."""
    client = app_anthropic_tool_use
    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-qwable-fast",
            "max_tokens": 1024,
            "tools": [{"name": "Bash", "input_schema": {}}],
            "messages": [{"role": "user", "content": "List directory"}],
            "stream": False,
        },
        headers={
            "x-api-key": "local",
            "anthropic-version": "2023-06-01",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["content"][0]["type"] == "tool_use"
    assert data["content"][0]["name"] == "Bash"
