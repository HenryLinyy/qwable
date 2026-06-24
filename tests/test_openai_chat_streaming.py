"""Tests for OpenAI Chat Completions streaming."""


def test_chat_streaming_final_answer(app_final_answer):
    client = app_final_answer
    response = client.post("/v1/chat/completions", json={
        "model": "qwable-chat",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    })
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert '"object": "chat.completion.chunk"' in body
    assert '"content": "Hello world"' in body
    assert "data: [DONE]" in body


def test_chat_streaming_tool_call(app_tool_call):
    client = app_tool_call
    response = client.post("/v1/chat/completions", json={
        "model": "qwable-fast",
        "messages": [{"role": "user", "content": "Read file"}],
        "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        "stream": True,
    })
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert '"tool_calls"' in body
    assert '"arguments": "{\\"path\\":\\"test.txt\\"}"' in body
    assert "data: [DONE]" in body
