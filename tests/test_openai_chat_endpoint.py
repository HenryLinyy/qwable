"""Tests for OpenAI Chat endpoint."""

import json


def test_chat_final_answer(app_final_answer):
    """POST /v1/chat/completions should return final answer."""
    client = app_final_answer
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwable-chat",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["content"] == "Hello world"
    assert set(data["usage"]) == {"prompt_tokens", "completion_tokens", "total_tokens"}


def test_chat_tool_call(app_tool_call):
    """POST /v1/chat/completions should return tool call."""
    client = app_tool_call
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwable-fast",
            "messages": [{"role": "user", "content": "Read file"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "read_file", "parameters": {}},
                }
            ],
            "stream": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["tool_calls"] is not None
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    arguments = data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, str)
    assert json.loads(arguments) == {"path": "test.txt"}
