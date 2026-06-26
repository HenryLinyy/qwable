"""Tests for OpenAI Responses streaming."""

import asyncio
import time
from fastapi.testclient import TestClient
from qwable.config import FusionConfig
from qwable.schemas import FusionAction
import qwable.server as server_mod


def test_responses_streaming(app_final_answer):
    """POST /v1/responses with stream=True should return SSE."""
    client = app_final_answer
    response = client.post(
        "/v1/responses",
        json={
            "model": "qwable-fast",
            "input": "Hello",
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "event: response.created" in body
    assert "event: response.completed" in body
    assert '"type": "response.output_text.delta"' in body


def test_responses_streaming_tool_call(app_tool_call):
    """POST /v1/responses with stream=True and tool call."""
    client = app_tool_call
    response = client.post(
        "/v1/responses",
        json={
            "model": "qwable-fast",
            "input": "Read file",
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "read_file", "parameters": {}},
                }
            ],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "event: response.output_item.added" in body
    assert '"type": "function_call"' in body


def test_responses_streaming_keepalive_while_execute_blocks_event_loop():
    """Streaming should emit keepalive while a sync model call blocks execute."""

    class BlockingCore:
        async def execute(self, task):
            time.sleep(1.2)
            return FusionAction(
                type="final_answer",
                text="slow answer",
                tool_name=None,
                tool_input=None,
                confidence=1.0,
                rationale_summary=None,
            )

    server_mod.config = FusionConfig(stream_keepalive_seconds=1)
    server_mod.fusion_core = BlockingCore()
    server_mod.global_lock = asyncio.Lock()

    client = TestClient(server_mod.app)
    response = client.post(
        "/v1/responses",
        json={
            "model": "qwable-heavy",
            "input": "slow streaming task",
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert ": keepalive" in body
    assert "event: response.completed" in body
