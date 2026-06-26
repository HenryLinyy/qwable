"""Shared test fixtures for Qwable Gateway."""

import asyncio
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from qwable.config import FusionConfig
import pytest


def _make_mock_fusion_core(
    type="final_answer",
    text="Hello world",
    tool_name=None,
    tool_input=None,
    confidence=1.0,
    rationale_summary=None,
    trace=None,
):
    """Helper to create a mocked FusionCore with a specific return value."""
    mock = MagicMock()
    result = MagicMock(
        type=type,
        text=text,
        tool_name=tool_name,
        tool_input=tool_input,
        confidence=confidence,
        rationale_summary=rationale_summary,
        trace=trace,
    )

    async def mock_execute(*args, **kwargs):
        return result

    mock.execute = mock_execute
    mock.__bool__ = lambda self: True
    return mock


def _make_test_client(mock_fusion_core):
    """Create a TestClient with initialized globals and a mocked fusion_core."""
    import qwable.server as server_mod

    config = FusionConfig()
    lock = asyncio.Lock()

    # Directly set module-level globals before creating the app
    server_mod.config = config
    server_mod.fusion_core = mock_fusion_core
    server_mod.global_lock = lock

    return TestClient(server_mod.app)


@pytest.fixture
def app_final_answer():
    """TestClient with fusion_core mocked to return final_answer."""
    mock = _make_mock_fusion_core(type="final_answer", text="Hello world")
    return _make_test_client(mock)


@pytest.fixture
def app_tool_call():
    """TestClient with fusion_core mocked to return tool_call."""
    mock = _make_mock_fusion_core(
        type="tool_call",
        text=None,
        tool_name="read_file",
        tool_input={"path": "test.txt"},
        confidence=0.9,
    )
    return _make_test_client(mock)


@pytest.fixture
def app_heavy_debug_answer():
    """TestClient with fusion_core mocked to return heavy-agent debug trace."""
    mock = _make_mock_fusion_core(
        type="final_answer",
        text="heavy answer",
        rationale_summary="heavy_resource_guard: Required 175.0GB exceeds limit 100.0GB",
        trace={
            "profile": "heavy-agent",
            "heavy_backend": "ds4",
            "fallback": None,
            "resource_guard": True,
            "reason": "Required 175.0GB exceeds limit 100.0GB",
        },
    )
    return _make_test_client(mock)


@pytest.fixture
def app_anthropic_final_answer():
    """TestClient with fusion_core mocked for Anthropic final answer."""
    mock = _make_mock_fusion_core(type="final_answer", text="Hello world")
    return _make_test_client(mock)


@pytest.fixture
def app_anthropic_tool_use():
    """TestClient with fusion_core mocked for Anthropic tool use."""
    mock = _make_mock_fusion_core(
        type="tool_call",
        text=None,
        tool_name="Bash",
        tool_input={"command": "ls"},
        confidence=0.9,
    )
    return _make_test_client(mock)
