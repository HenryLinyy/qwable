"""Tests for global lock mechanism."""

import asyncio
import pytest
import time
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def slow_client():
    """TestClient where fusion_core.execute takes 0.3s to simulate a busy server."""
    import qwable.server as server_mod

    config = __import__("qwable.config", fromlist=["FusionConfig"]).FusionConfig()
    __import__("qwable.fusion_core", fromlist=["FusionCore"]).FusionCore(config)
    lock = asyncio.Lock()

    mock = MagicMock()

    async def async_execute(*args, **kwargs):
        await asyncio.sleep(0.3)
        return MagicMock(
            type="final_answer",
            text="Hello",
            tool_name=None,
            tool_input=None,
            confidence=1.0,
            rationale_summary=None,
        )

    mock.execute = async_execute
    mock.__bool__ = lambda self: True

    server_mod.config = config
    server_mod.fusion_core = mock
    server_mod.global_lock = lock

    return TestClient(server_mod.app)


@pytest.mark.asyncio
async def test_global_lock_busy(slow_client):
    """Second concurrent request should get 429."""
    import httpx
    import qwable.server as server_mod

    # Use ASGITransport for httpx>=0.28, where the old app shortcut was removed.
    transport = httpx.ASGITransport(app=server_mod.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with asyncio.TaskGroup() as tg:
            t1 = tg.create_task(
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "qwable-chat",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": False,
                    },
                )
            )
            # Small delay to ensure request 1 starts first
            await asyncio.sleep(0.05)
            t2 = tg.create_task(
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "qwable-chat",
                        "messages": [{"role": "user", "content": "Hello again"}],
                        "stream": False,
                    },
                )
            )

        r1 = t1.result()
        r2 = t2.result()

        assert r1.status_code == 200
        assert r2.status_code == 429


def test_global_lock_release(slow_client):
    """After first request completes, second should succeed."""
    client = slow_client
    response1 = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwable-chat",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )
    assert response1.status_code == 200

    # Wait for lock release
    time.sleep(0.5)

    response2 = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwable-chat",
            "messages": [{"role": "user", "content": "Hello again"}],
            "stream": False,
        },
    )
    assert response2.status_code == 200
