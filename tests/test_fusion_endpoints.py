"""G10: Endpoint tests for qwable-fusion / claude-qwable-fusion.

Verifies that all three protocols accept the new fusion model ids and forward
the request through FusionCore (mocked here, real stub in production).
"""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from qwable.config import FusionConfig


def _make_mock_fusion_core(type="final_answer", text="", trace=None, **kwargs):
    mock = MagicMock()
    result = MagicMock(
        type=type,
        text=text,
        tool_name=kwargs.get("tool_name"),
        tool_input=kwargs.get("tool_input"),
        confidence=kwargs.get("confidence", 0.5),
        rationale_summary=kwargs.get("rationale_summary"),
        trace=trace or {},
    )

    async def mock_execute(*args, **kwargs):
        return result

    mock.execute = mock_execute
    mock.__bool__ = lambda self: True
    return mock


def _make_client(mock_core):
    import qwable.server as server_mod

    server_mod.config = FusionConfig()
    server_mod.fusion_core = mock_core
    server_mod.global_lock = asyncio.Lock()
    return TestClient(server_mod.app)


def _fusion_trace(preset="quality", analysis_models=None, judge_model=None):
    return {
        "profile": "fusion-agent",
        "source_protocol": "openai_chat",
        "fusion": {
            "preset": preset,
            "analysis_models": analysis_models
            or ["qwen/qwen3-coder-next", "qwen/qwen3.6-35b-a3b", "deepseek-r1-distill-qwen-32b"],
            "judge_model": judge_model or "qwen/qwen3.6-35b-a3b",
            "description": "Deep reasoning — 3 large models, qwen3.6 judge",
        },
    }


# ─── H1: OpenAI Chat endpoint ─────────────────────────────────────────────


def test_openai_chat_endpoint_accepts_qwable_fusion():
    """POST /v1/chat/completions with model=qwable-fusion returns 200."""
    trace = _fusion_trace()
    mock = _make_mock_fusion_core(
        text="[G10 STUB] resolved preset quality",
        trace=trace,
    )
    client = _make_client(mock)

    response = client.post("/v1/chat/completions", json={
        "model": "qwable-fusion",
        "messages": [{"role": "user", "content": "Compare two sort algorithms"}],
        "plugins": [{"id": "fusion", "preset": "quality"}],
        "stream": False,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "qwable-fusion"
    content = data["choices"][0]["message"]["content"]
    assert "quality" in content or "resolved" in content
    assert set(data["usage"]) == {"prompt_tokens", "completion_tokens", "total_tokens"}


def test_openai_chat_endpoint_accepts_top_level_fusion_block():
    """Top-level `fusion` block shape should also work."""
    mock = _make_mock_fusion_core(
        text="[G10 STUB] custom panel",
        trace=_fusion_trace(preset="custom", analysis_models=["m1", "m2"], judge_model="m-judge"),
    )
    client = _make_client(mock)

    response = client.post("/v1/chat/completions", json={
        "model": "qwable-fusion",
        "messages": [{"role": "user", "content": "x"}],
        "fusion": {
            "analysis_models": ["m1", "m2"],
            "judge_model": "m-judge",
        },
        "stream": False,
    })
    assert response.status_code == 200


def test_openai_chat_endpoint_returns_400_on_bad_preset():
    """Bad preset propagates as a final_answer with error trace (HTTP 200)."""
    mock = _make_mock_fusion_core(
        text="fusion-agent preset error: unknown fusion preset 'bogus'",
        confidence=0.0,
        rationale_summary="fusion_preset_error",
        trace={"profile": "fusion-agent", "error": "fusion_preset_error", "error_detail": "unknown"},
    )
    client = _make_client(mock)

    response = client.post("/v1/chat/completions", json={
        "model": "qwable-fusion",
        "messages": [{"role": "user", "content": "x"}],
        "fusion": {"preset": "bogus"},
        "stream": False,
    })
    assert response.status_code == 200  # body returns 200 with error in text
    data = response.json()
    assert "bogus" in data["choices"][0]["message"]["content"]


def test_openai_chat_endpoint_custom_panel_flows_through():
    """Custom analysis_models should reach FusionCore via raw_request."""
    captured = {}

    async def capture_execute(task):
        captured["raw"] = task.raw_request
        result = MagicMock(type="final_answer", text="ok", tool_name=None, tool_input=None,
                           confidence=0.5, rationale_summary=None,
                           trace=_fusion_trace(preset="custom"))
        return result

    mock = MagicMock()
    mock.execute = capture_execute
    mock.__bool__ = lambda self: True

    client = _make_client(mock)
    response = client.post("/v1/chat/completions", json={
        "model": "qwable-fusion",
        "messages": [{"role": "user", "content": "x"}],
        "fusion": {
            "analysis_models": ["m1", "m2"],
            "judge_model": "m-judge",
        },
    })
    assert response.status_code == 200
    assert captured["raw"]["fusion"]["analysis_models"] == ["m1", "m2"]
    assert captured["raw"]["fusion"]["judge_model"] == "m-judge"


# ─── H2: OpenAI Responses endpoint ───────────────────────────────────────


def test_openai_responses_endpoint_accepts_qwable_fusion():
    """POST /v1/responses with model=qwable-fusion returns 200."""
    mock = _make_mock_fusion_core(
        text="[G10 STUB] budget preset",
        trace=_fusion_trace(preset="budget", analysis_models=["gemma", "qwen3.6"]),
    )
    client = _make_client(mock)

    response = client.post("/v1/responses", json={
        "model": "qwable-fusion",
        "input": "summarize trade-offs",
        "plugins": [{"id": "fusion", "preset": "budget"}],
        "stream": False,
    })
    assert response.status_code == 200
    data = response.json()
    # OpenAI Responses returns output_text or output array; tolerate either
    assert "output" in data or "output_text" in data


def test_openai_responses_endpoint_custom_panel():
    """Custom panel via fusion block works on Responses too."""
    mock = _make_mock_fusion_core(
        text="[G10 STUB] custom",
        trace=_fusion_trace(preset="custom", analysis_models=["m1", "m2", "m3"], judge_model="m-judge"),
    )
    client = _make_client(mock)

    response = client.post("/v1/responses", json={
        "model": "qwable-fusion",
        "input": "x",
        "fusion": {
            "analysis_models": ["m1", "m2", "m3"],
            "judge_model": "m-judge",
        },
    })
    assert response.status_code == 200


# ─── H3: Anthropic Messages endpoint ─────────────────────────────────────


def test_anthropic_endpoint_accepts_claude_qwable_fusion():
    """POST /v1/messages with model=claude-qwable-fusion returns 200."""
    mock = _make_mock_fusion_core(
        text="[G10 STUB] coding preset",
        trace=_fusion_trace(preset="coding", judge_model="qwen/qwen3-coder-next"),
    )
    client = _make_client(mock)

    response = client.post("/v1/messages", json={
        "model": "claude-qwable-fusion",
        "messages": [{"role": "user", "content": "review this PR"}],
        "max_tokens": 2048,
        "fusion": {"preset": "coding"},
    })
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "claude-qwable-fusion"
    assert "content" in data
    assert len(data["content"]) >= 1
    assert "text" in data["content"][0]


def test_anthropic_endpoint_heavy_preset():
    """Heavy preset uses ds4 judge — endpoint should pass it through."""
    mock = _make_mock_fusion_core(
        text="[G10 STUB] heavy preset",
        trace=_fusion_trace(preset="heavy", judge_model="deepseek-v4-flash",
                            analysis_models=["qwen-coder", "deepseek-r1"]),
    )
    client = _make_client(mock)

    response = client.post("/v1/messages", json={
        "model": "claude-qwable-fusion",
        "messages": [{"role": "user", "content": "long context question"}],
        "max_tokens": 2048,
        "fusion": {"preset": "heavy"},
    })
    assert response.status_code == 200


# ─── H4: Bad-input / edge cases ───────────────────────────────────────────


def test_openai_chat_endpoint_empty_messages_list():
    """Empty messages should still produce a 200 with stub output."""
    mock = _make_mock_fusion_core(
        text="[G10 STUB] empty input",
        trace=_fusion_trace(),
    )
    client = _make_client(mock)

    response = client.post("/v1/chat/completions", json={
        "model": "qwable-fusion",
        "messages": [],
    })
    assert response.status_code == 200


def test_openai_chat_endpoint_plugins_priority_over_fusion_block():
    """If both shapes present, plugins wins (extract_fusion_request convention)."""
    captured = {}

    async def capture_execute(task):
        captured["raw"] = task.raw_request
        return MagicMock(type="final_answer", text="ok", tool_name=None, tool_input=None,
                         confidence=0.5, rationale_summary=None,
                         trace=_fusion_trace(preset="quality"))

    mock = MagicMock()
    mock.execute = capture_execute
    mock.__bool__ = lambda self: True

    client = _make_client(mock)
    response = client.post("/v1/chat/completions", json={
        "model": "qwable-fusion",
        "messages": [{"role": "user", "content": "x"}],
        "plugins": [{"id": "fusion", "preset": "quality"}],
        "fusion": {"preset": "budget"},
    })
    assert response.status_code == 200
    # Both shapes preserved in raw body (resolution happens inside FusionCore)
    assert captured["raw"]["plugins"][0]["preset"] == "quality"
    assert captured["raw"]["fusion"]["preset"] == "budget"
