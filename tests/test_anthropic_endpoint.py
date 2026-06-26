"""Tests for Anthropic Messages endpoint."""


def test_anthropic_final_answer(app_anthropic_final_answer):
    """POST /v1/messages should return final answer."""
    client = app_anthropic_final_answer
    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-qwable-fast",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "message"
    assert data["stop_reason"] == "end_turn"
    assert data["content"][0]["type"] == "text"
    assert data["content"][0]["text"] == "Hello world"


def test_anthropic_tool_use(app_anthropic_tool_use):
    """POST /v1/messages should return tool use."""
    client = app_anthropic_tool_use
    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-qwable-fast",
            "messages": [{"role": "user", "content": "List files"}],
            "max_tokens": 100,
            "tools": [
                {"type": "function", "function": {"name": "Bash", "parameters": {}}}
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["stop_reason"] == "tool_use"
    assert data["content"][0]["type"] == "tool_use"
