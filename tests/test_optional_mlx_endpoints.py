"""Endpoint contract tests for optional MLX aliases."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from qwable.schemas import FusionAction


class CaptureFusionCore:
    def __init__(self):
        self.tasks = []

    async def execute(self, task):
        self.tasks.append(task)
        return FusionAction(
            type="final_answer",
            text="mlx ok",
            tool_name=None,
            tool_input=None,
            confidence=1.0,
            rationale_summary=None,
        )


def _client_with_capture():
    import qwable.server as server_mod

    core = CaptureFusionCore()
    server_mod.fusion_core = core
    server_mod.global_lock = asyncio.Lock()
    server_mod.config = None
    return TestClient(server_mod.app), core


@pytest.mark.parametrize(
    ("model", "expected_profile"),
    [
        ("qwable-agentic-mlx", "agentic-mlx"),
        ("qwable-formatter-mlx", "formatter-mlx"),
    ],
)
def test_openai_responses_routes_optional_mlx_aliases(model, expected_profile):
    client, core = _client_with_capture()

    response = client.post(
        "/v1/responses",
        json={
            "model": model,
            "input": "hello",
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert core.tasks[-1].profile == expected_profile
    assert response.json()["output"][0]["text"] == "mlx ok"


@pytest.mark.parametrize(
    ("model", "expected_profile"),
    [
        ("qwable-agentic-mlx", "agentic-mlx"),
        ("qwable-formatter-mlx", "formatter-mlx"),
    ],
)
def test_openai_chat_routes_optional_mlx_aliases(model, expected_profile):
    client, core = _client_with_capture()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert core.tasks[-1].profile == expected_profile
    assert response.json()["choices"][0]["message"]["content"] == "mlx ok"


@pytest.mark.parametrize(
    ("model", "expected_profile"),
    [
        ("claude-qwable-agentic-mlx", "agentic-mlx"),
        ("claude-qwable-formatter-mlx", "formatter-mlx"),
    ],
)
def test_anthropic_messages_routes_optional_mlx_aliases(model, expected_profile):
    client, core = _client_with_capture()

    response = client.post(
        "/v1/messages",
        json={
            "model": model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
        headers={
            "x-api-key": "local",
            "anthropic-version": "2023-06-01",
        },
    )

    assert response.status_code == 200
    assert core.tasks[-1].profile == expected_profile
    assert response.json()["content"][0]["text"] == "mlx ok"
