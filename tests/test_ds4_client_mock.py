"""Tests for ds4 client with mocked HTTP."""

from unittest.mock import MagicMock
from qwable.models import DS4Client, OllamaClient


def test_ds4_health_ok():
    """DS4Client.health should probe /v1/models and return True on 200."""
    client = DS4Client("http://127.0.0.1:8000/v1")
    client.client = MagicMock()
    client.client.get.return_value.status_code = 200
    client.client.get.return_value.json.return_value = {"object": "list", "data": []}
    assert client.health() is True
    client.client.get.assert_called_once_with("http://127.0.0.1:8000/v1/models")


def test_ds4_health_fail():
    """DS4Client.health should return False on non-200."""
    client = DS4Client("http://127.0.0.1:8000/v1")
    client.client = MagicMock()
    client.client.get.return_value.status_code = 503
    assert client.health() is False


def test_ds4_health_exception():
    """DS4Client.health should return False on exception."""
    client = DS4Client("http://127.0.0.1:8000/v1")
    client.client = MagicMock()
    client.client.get.side_effect = Exception("Connection refused")
    assert client.health() is False


def test_ds4_chat_completion():
    """DS4Client.chat_completion should return response."""
    client = DS4Client("http://127.0.0.1:8000")
    client.client = MagicMock()
    mock_response = {"choices": [{"message": {"content": "OK"}}]}
    client.client.post.return_value.json.return_value = mock_response
    result = client.chat_completion(model="deepseek-v4-flash", messages=[{"role": "user", "content": "hi"}])
    assert result["choices"][0]["message"]["content"] == "OK"


def test_ollama_unload_models_uses_native_keep_alive_zero():
    """OllamaClient should unload unique model names through the native Ollama API."""
    client = OllamaClient("http://127.0.0.1:11434/v1")
    client.client = MagicMock()
    client.unload_models(["qwen3-coder:30b", "deepseek-r1:32b", "qwen3-coder:30b", ""])

    assert client.client.post.call_count == 2
    client.client.post.assert_any_call(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": "qwen3-coder:30b",
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
        },
    )
    client.client.post.assert_any_call(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": "deepseek-r1:32b",
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
        },
    )
